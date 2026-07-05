"""Sparse local Forward-Forward MoE with an independent local byte decoder."""

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import torch

torch.set_grad_enabled(False)


def sigmoid(x):
    return torch.sigmoid(x)


def rmsnorm(x, eps=1e-6):
    scale = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    return x * scale.to(x.dtype)


def prefix_mean(x):
    total = x.float().cumsum(1) - x.float()
    den = torch.arange(x.shape[1], device=x.device, dtype=torch.float32).clamp_min_(1)
    value = (total / den[None, :, None]).to(x.dtype)
    value[:, 0].zero_()
    return value


def plasticity_(weight, local_derivative, rate, clip=4.0):
    derivative32 = local_derivative.float()
    normalizer = derivative32.square().mean().sqrt().clamp_min_(1e-6)
    change32 = derivative32.div(normalizer).clamp_(-clip, clip)
    change32.nan_to_num_(nan=0.0, posinf=clip, neginf=-clip)
    weight.add_(change32.to(weight.dtype), alpha=-rate)


@dataclass
class Config:
    vocab: int = 512
    active_tokens: int = 264
    width: int = 64
    depth: int = 2
    experts: int = 4
    expert_hidden: int = 128
    threshold: float = 1.0
    rate: float = 1e-3
    router_rate: float = 2e-4
    decoder_rate: float = 2e-4
    seed: int = 20260629

    @classmethod
    def nominal_4b(cls, seed=20260629):
        return cls(vocab=32768, active_tokens=264, width=1024, depth=20,
                   experts=24, expert_hidden=4096, threshold=1.0,
                   rate=1e-3, router_rate=2e-4, decoder_rate=2e-4,
                   seed=seed)

    def backbone_parameter_count(self):
        emb = self.vocab * self.width
        per_expert = (self.width * self.expert_hidden +
                      self.expert_hidden * self.width +
                      self.expert_hidden + self.width)
        per_block = (2 * self.width * self.width + self.width +
                     self.width * self.experts + self.experts +
                     self.experts * per_expert)
        return emb + self.depth * per_block

    def parameter_count(self):
        return self.backbone_parameter_count() + self.width * self.active_tokens + self.active_tokens


class SparseLocalBlock:
    def __init__(self, cfg, device, dtype, generator):
        d, h, e = cfg.width, cfg.expert_hidden, cfg.experts
        self.cfg = cfg
        self.mix = torch.randn(2*d, d, device=device, dtype=dtype, generator=generator).mul_(d ** -0.5)
        self.bias = torch.zeros(d, device=device, dtype=dtype)
        self.router = torch.randn(d, e, device=device, dtype=dtype, generator=generator).mul_(d ** -0.5)
        self.router_bias = torch.zeros(e, device=device, dtype=torch.float32)
        self.up = torch.randn(e, d, h, device=device, dtype=dtype, generator=generator).mul_(d ** -0.5)
        self.down = torch.randn(e, h, d, device=device, dtype=dtype, generator=generator).mul_(h ** -0.5)
        self.up_bias = torch.zeros(e, h, device=device, dtype=dtype)
        self.down_bias = torch.zeros(e, d, device=device, dtype=dtype)

    def _stem(self, x):
        q = torch.cat((x, prefix_mean(x)), -1)
        pre = torch.matmul(q, self.mix).add_(self.bias)
        s = sigmoid(pre)
        stem = pre * s
        route = (torch.matmul(stem, self.router).float() + self.router_bias).argmax(-1)
        return q, pre, s, stem, route

    def _expert_side(self, stem_rows, expert_id):
        pre = torch.matmul(stem_rows, self.up[expert_id]).add_(self.up_bias[expert_id])
        s = sigmoid(pre)
        hidden = pre * s
        out = stem_rows + torch.matmul(hidden, self.down[expert_id]).add_(self.down_bias[expert_id])
        goodness = out.float().square().mean(-1)
        return pre, s, hidden, out, goodness

    def learn(self, positive, negative):
        qp, pp, sp, hp, rp = self._stem(positive)
        qn, pn, sn, hn, rn = self._stem(negative)
        shape = hp.shape
        hp_flat, hn_flat = hp.reshape(-1, shape[-1]), hn.reshape(-1, shape[-1])
        rp_flat, rn_flat = rp.reshape(-1), rn.reshape(-1)
        outp, outn = torch.empty_like(hp_flat), torch.empty_like(hn_flat)
        dhp, dhn = torch.zeros_like(hp_flat), torch.zeros_like(hn_flat)
        losses, correct, observations = [], 0.0, 0
        for expert in range(self.cfg.experts):
            ip = torch.nonzero(rp_flat == expert, as_tuple=False).flatten()
            inn = torch.nonzero(rn_flat == expert, as_tuple=False).flatten()
            if ip.numel() == 0 and inn.numel() == 0:
                continue
            grad_up = torch.zeros_like(self.up[expert]); grad_down = torch.zeros_like(self.down[expert])
            grad_ub = torch.zeros_like(self.up_bias[expert]); grad_db = torch.zeros_like(self.down_bias[expert])
            if ip.numel():
                ap, sap, vp, op, gp = self._expert_side(hp_flat[ip], expert)
                alpha = -sigmoid(self.cfg.threshold - gp).div_(gp.numel())
                dup = (alpha[:, None] * (2.0 / self.cfg.width) * op.float()).to(hp.dtype)
                dv = torch.matmul(dup, self.down[expert].T)
                da = dv * sap * (1 + ap * (1 - sap))
                grad_down.add_(torch.matmul(vp.T, dup)); grad_up.add_(torch.matmul(hp_flat[ip].T, da))
                grad_db.add_(dup.sum(0)); grad_ub.add_(da.sum(0))
                dhp[ip] = dup + torch.matmul(da, self.up[expert].T); outp[ip] = op
                losses.append(torch.nn.functional.softplus(self.cfg.threshold - gp).mean())
                correct += float((gp > self.cfg.threshold).sum()); observations += gp.numel()
                reward = sigmoid(self.cfg.threshold - gp).to(hp.dtype)
                self.router[:, expert].add_(torch.matmul(hp_flat[ip].T, reward[:, None]).squeeze(1) / ip.numel(),
                                            alpha=self.cfg.router_rate)
            if inn.numel():
                an, san, vn, on, gn = self._expert_side(hn_flat[inn], expert)
                alpha = sigmoid(gn - self.cfg.threshold).div_(gn.numel())
                dun = (alpha[:, None] * (2.0 / self.cfg.width) * on.float()).to(hn.dtype)
                dv = torch.matmul(dun, self.down[expert].T)
                da = dv * san * (1 + an * (1 - san))
                grad_down.add_(torch.matmul(vn.T, dun)); grad_up.add_(torch.matmul(hn_flat[inn].T, da))
                grad_db.add_(dun.sum(0)); grad_ub.add_(da.sum(0))
                dhn[inn] = dun + torch.matmul(da, self.up[expert].T); outn[inn] = on
                losses.append(torch.nn.functional.softplus(gn - self.cfg.threshold).mean())
                correct += float((gn < self.cfg.threshold).sum()); observations += gn.numel()
                reward = -sigmoid(gn - self.cfg.threshold).to(hn.dtype)
                self.router[:, expert].add_(torch.matmul(hn_flat[inn].T, reward[:, None]).squeeze(1) / inn.numel(),
                                            alpha=self.cfg.router_rate)
            plasticity_(self.down[expert], grad_down, self.cfg.rate)
            plasticity_(self.up[expert], grad_up, self.cfg.rate)
            plasticity_(self.down_bias[expert], grad_db, self.cfg.rate)
            plasticity_(self.up_bias[expert], grad_ub, self.cfg.rate)
        counts = torch.bincount(torch.cat((rp_flat, rn_flat)), minlength=self.cfg.experts).float()
        self.router_bias.add_((counts.mean() - counts) / counts.sum().clamp_min(1), alpha=self.cfg.router_rate)
        dhp, dhn = dhp.reshape(shape), dhn.reshape(shape)
        dpp = dhp * sp * (1 + pp * (1 - sp)); dpn = dhn * sn * (1 + pn * (1 - sn))
        grad_mix = torch.matmul(qp.reshape(-1, qp.shape[-1]).T, dpp.reshape(-1, dpp.shape[-1]))
        grad_mix.add_(torch.matmul(qn.reshape(-1, qn.shape[-1]).T, dpn.reshape(-1, dpn.shape[-1])))
        plasticity_(self.mix, grad_mix, self.cfg.rate)
        plasticity_(self.bias, dpp.sum((0, 1)) + dpn.sum((0, 1)), self.cfg.rate)
        mean_loss = torch.stack(losses).mean().item() if losses else 0.0
        return rmsnorm(outp.reshape(shape)), rmsnorm(outn.reshape(shape)), mean_loss, correct / max(observations, 1)

    def forward(self, x, return_goodness=False):
        _, _, _, stem, route = self._stem(x)
        shape = stem.shape; flat, ids = stem.reshape(-1, shape[-1]), route.reshape(-1)
        out = torch.empty_like(flat)
        for expert in range(self.cfg.experts):
            idx = torch.nonzero(ids == expert, as_tuple=False).flatten()
            if idx.numel():
                _, _, _, value, _ = self._expert_side(flat[idx], expert); out[idx] = value
        out = out.reshape(shape)
        goodness = out.float().square().mean(-1); normalized = rmsnorm(out)
        return (normalized, goodness) if return_goodness else normalized

    def forward_one(self, x, prefix):
        q = torch.cat((x, prefix), -1)
        pre = torch.matmul(q, self.mix).add_(self.bias)
        stem = pre * sigmoid(pre)
        expert = int((torch.matmul(stem, self.router).float() + self.router_bias).argmax(-1).item())
        _, _, _, out, _ = self._expert_side(stem, expert)
        return rmsnorm(out)

    def tensors(self):
        return {"mix": self.mix, "bias": self.bias, "router": self.router,
                "router_bias": self.router_bias, "up": self.up, "down": self.down,
                "up_bias": self.up_bias, "down_bias": self.down_bias}


class LocalByteDecoder:
    def __init__(self, cfg, device, dtype, generator):
        self.cfg = cfg
        self.weight = torch.randn(cfg.width, cfg.active_tokens, device=device, dtype=dtype,
                                  generator=generator).mul_(cfg.width ** -0.5)
        self.bias = torch.zeros(cfg.active_tokens, device=device, dtype=torch.float32)

    def logits(self, hidden):
        return torch.matmul(hidden.float(), self.weight.float()) + self.bias

    def learn(self, hidden, ids):
        x = hidden[:, :-1].reshape(-1, self.cfg.width).float()
        targets = ids[:, 1:].reshape(-1)
        keep = targets < self.cfg.active_tokens
        x, targets = x[keep], targets[keep]
        logits = self.logits(x); logits.sub_(logits.max(-1, keepdim=True).values)
        exp = logits.exp(); probabilities = exp / exp.sum(-1, keepdim=True).clamp_min_(1e-12)
        row = torch.arange(targets.numel(), device=targets.device)
        nll = -torch.log(probabilities[row, targets].clamp_min_(1e-12)).mean()
        accuracy = (probabilities.argmax(-1) == targets).float().mean()
        reconstruction_mse = (probabilities.square().sum(-1) -
                              2 * probabilities[row, targets] + 1).mean() / self.cfg.active_tokens
        delta = probabilities
        delta[row, targets] -= 1.0
        delta.div_(max(targets.numel(), 1))
        plasticity_(self.weight, torch.matmul(x.T, delta), self.cfg.decoder_rate)
        plasticity_(self.bias, delta.sum(0), self.cfg.decoder_rate)
        return float(nll), float(accuracy), float(reconstruction_mse)

    def tensors(self):
        return {"decoder_weight": self.weight, "decoder_bias": self.bias}


class LocalDecoderMoE4B:
    def __init__(self, cfg, device):
        self.cfg, self.device = cfg, device
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        gen = torch.Generator(device=device).manual_seed(cfg.seed)
        self.embedding = rmsnorm(torch.randn(cfg.vocab, cfg.width, device=device, dtype=dtype, generator=gen))
        self.blocks = [SparseLocalBlock(cfg, device, dtype, gen) for _ in range(cfg.depth)]
        self.decoder = LocalByteDecoder(cfg, device, dtype, gen)
        assert all(not tensor.requires_grad for tensor in self.all_tensors())

    def all_tensors(self):
        yield self.embedding
        for block in self.blocks: yield from block.tensors().values()
        yield from self.decoder.tensors().values()

    def learn_pair(self, positive_ids, negative_ids):
        pos, neg = self.embedding[positive_ids], self.embedding[negative_ids]
        losses, accuracies = [], []
        for block in self.blocks:
            pos, neg, loss, acc = block.learn(pos, neg)
            losses.append(loss); accuracies.append(acc)
        decoder_nll, decoder_accuracy, decoder_mse = self.decoder.learn(pos, positive_ids)
        return (sum(losses)/len(losses), sum(accuracies)/len(accuracies),
                decoder_nll, decoder_accuracy, decoder_mse)

    def represent(self, ids):
        x = self.embedding[ids]
        for block in self.blocks: x = block.forward(x)
        return x

    def energy(self, ids):
        x = self.embedding[ids]
        total = torch.zeros(ids.shape, device=ids.device, dtype=torch.float32)
        for block in self.blocks:
            x, goodness = block.forward(x, return_goodness=True); total.add_(goodness)
        return total / len(self.blocks)

    def token_nll(self, ids):
        hidden = self.represent(ids)
        logits = self.decoder.logits(hidden[:, :-1])
        log_z = torch.logsumexp(logits, -1)
        target = ids[:, 1:].clamp(0, self.cfg.active_tokens - 1)
        chosen = logits.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        return log_z - chosen

    def new_cache(self):
        return [{"sum": torch.zeros(1, self.cfg.width, device=self.device), "count": 0}
                for _ in self.blocks]

    def forward_token(self, token_id, cache):
        x = self.embedding[int(token_id)].reshape(1, -1)
        for index, block in enumerate(self.blocks):
            state = cache[index]
            prefix = (state["sum"] / state["count"]).to(x.dtype) if state["count"] else torch.zeros_like(x)
            old_x = x
            x = block.forward_one(x, prefix)
            state["sum"].add_(old_x.float()); state["count"] += 1
        return self.decoder.logits(x).squeeze(0)

    def generate(self, prompt_ids, max_new_tokens, generator, temperature=0.8, top_k=40):
        cache = self.new_cache(); logits = None
        for token in prompt_ids[-128:]: logits = self.forward_token(token, cache)
        result = []
        for _ in range(max_new_tokens):
            sample_logits = logits.clone(); sample_logits[:3] = -float("inf")
            values, indices = torch.topk(sample_logits, min(top_k, sample_logits.numel()))
            probabilities = torch.softmax(values / temperature, 0)
            token = int(indices[torch.multinomial(probabilities, 1, generator=generator)].item())
            if token == 3: break
            result.append(token); logits = self.forward_token(token, cache)
        return result

    def save_sharded(self, directory, step):
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text(json.dumps(asdict(self.cfg), indent=2), encoding="utf-8")
        torch.save({"step": step, "embedding": self.embedding.cpu()}, directory / "embedding.pt")
        self.embedding = self.embedding.to(self.device)
        for i, block in enumerate(self.blocks):
            torch.save({k: v.cpu() for k, v in block.tensors().items()}, directory / f"block_{i:02d}.pt")
        torch.save({k: v.cpu() for k, v in self.decoder.tensors().items()}, directory / "decoder.pt")
        self.decoder.weight = self.decoder.weight.to(self.device); self.decoder.bias = self.decoder.bias.to(self.device)

    @classmethod
    def load_sharded(cls, directory, device):
        directory = Path(directory)
        cfg = Config(**json.loads((directory / "config.json").read_text(encoding="utf-8")))
        model = cls(cfg, device)
        model.embedding = torch.load(directory / "embedding.pt", map_location=device,
                                     weights_only=True)["embedding"].to(device)
        for i, block in enumerate(model.blocks):
            state = torch.load(directory / f"block_{i:02d}.pt", map_location=device, weights_only=True)
            for key, value in state.items(): setattr(block, key, value.to(device))
        decoder_state = torch.load(directory / "decoder.pt", map_location=device, weights_only=True)
        model.decoder.weight = decoder_state["decoder_weight"].to(device)
        model.decoder.bias = decoder_state["decoder_bias"].to(device)
        return model


def corrupt(ids, vocab, generator, probability=0.15):
    mask = torch.rand(ids.shape, device=ids.device, generator=generator) < probability
    replacement = torch.randint(0, vocab, ids.shape, device=ids.device, generator=generator)
    result = torch.where(mask, replacement, ids)
    result[:, -1] = (result[:, -1] + 1) % vocab
    return result
