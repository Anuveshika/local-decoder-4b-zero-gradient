# CPU and edge deployment

## Standard local CPU

The full native checkpoint is approximately 8.2 GB in FP16 before PyTorch
overhead. Use a 64-bit CPU host with at least 12–16 GB free RAM, SSD storage and
Python 3.12. Generation is intentionally single-process and capped at four CPU
threads by default.

1. Download the complete `checkpoint/` directory from the successful Kaggle
   output.
2. Verify it contains `config.json`, `embedding.pt`, `decoder.pt` and
   `block_00.pt` through `block_19.pt`.
3. Install `torch==2.6.0` in a clean environment.
4. Run:

```bash
python -m src.inference_cpu \
  --checkpoint /models/local-decoder-4b/checkpoint \
  --prompt "Write one sentence about local predictive coding." \
  --max-new-bytes 64
```

`inference_cpu.py` constructs the object with `__new__` and directly attaches
checkpoint tensors. It does not allocate a second randomly initialized 4B
model. It uses the same cached one-token causal path as MT-Bench generation.

## Edge-class devices

The phrase “edge deployment” must not hide the physical footprint. A 4.105B
model needs approximately:

- 8.2 GB for FP16 weights;
- 4.1 GB for int8 weights before metadata and runtime workspace;
- more memory for caches, temporary dequantization and the Python runtime.

Consequently, the complete model is unsuitable for microcontrollers,
low-memory phones and typical Raspberry Pi configurations. A workstation-class
edge appliance with at least 8 GB available RAM may host an optimized int8
runtime.

Create a shardwise symmetric-int8 storage export:

```bash
python scripts/quantize_checkpoint_int8.py \
  --checkpoint /models/local-decoder-4b/checkpoint \
  --output /models/local-decoder-4b/checkpoint-int8
```

The export stores an int8 tensor and scalar scale for each source tensor. An
actual low-latency deployment should integrate these shards with a native
int8 GEMM runtime and preserve top-1 expert routing. The supplied exporter is a
portable storage conversion, not an optimized inference engine.

## Operational limitations

- Context length is 128 byte tokens.
- Output is byte-level and may contain malformed or low-quality text.
- Executed HellaSwag and PIQA scores are near chance.
- No safety alignment, factuality guarantee or production service hardening is
  claimed.
- Do not expose this model as a reliable public assistant.

