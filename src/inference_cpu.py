"""Memory-conscious CPU loader and conversational generation CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .local_decoder_4b import Config, LocalByteDecoder, LocalDecoderMoE4B, SparseLocalBlock

BYTE_OFFSET = 4
USER, ASSISTANT, TURN_END = 261, 262, 263


def encode(text: str) -> list[int]:
    return [byte + BYTE_OFFSET for byte in text.encode("utf-8", errors="replace")]


def decode(ids: list[int]) -> str:
    raw = bytes(token - BYTE_OFFSET for token in ids if BYTE_OFFSET <= token < 260)
    return raw.decode("utf-8", errors="replace")


def load_without_random_initialization(checkpoint: Path, device: torch.device) -> LocalDecoderMoE4B:
    """Load shards directly, avoiding a second 4B random model allocation."""
    checkpoint = Path(checkpoint)
    cfg = Config(**json.loads((checkpoint / "config.json").read_text(encoding="utf-8")))
    model = LocalDecoderMoE4B.__new__(LocalDecoderMoE4B)
    model.cfg, model.device = cfg, device
    model.embedding = torch.load(checkpoint / "embedding.pt", map_location=device,
                                 weights_only=True)["embedding"].to(device)
    model.blocks = []
    for index in range(cfg.depth):
        state = torch.load(checkpoint / f"block_{index:02d}.pt", map_location=device,
                           weights_only=True)
        block = SparseLocalBlock.__new__(SparseLocalBlock)
        block.cfg = cfg
        for key, value in state.items():
            setattr(block, key, value.to(device))
        model.blocks.append(block)
    state = torch.load(checkpoint / "decoder.pt", map_location=device, weights_only=True)
    decoder = LocalByteDecoder.__new__(LocalByteDecoder)
    decoder.cfg = cfg
    decoder.weight = state["decoder_weight"].to(device)
    decoder.bias = state["decoder_bias"].to(device)
    model.decoder = decoder
    assert all(not tensor.requires_grad for tensor in model.all_tensors())
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-bytes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260629)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = torch.device("cpu")
    model = load_without_random_initialization(args.checkpoint, device)
    prompt = [2, USER, *encode(args.prompt), TURN_END, ASSISTANT]
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    output = model.generate(prompt, args.max_new_bytes, generator)
    print(decode(output))


if __name__ == "__main__":
    main()

