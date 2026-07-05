"""Shardwise symmetric int8 storage export for edge-runtime integration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import torch


def quantize_tensor(value: torch.Tensor) -> dict:
    value32 = value.float()
    scale = float(value32.abs().max().clamp_min(1e-12) / 127.0)
    quantized = torch.round(value32 / scale).clamp(-127, 127).to(torch.int8)
    return {"q": quantized, "scale": scale, "shape": list(value.shape),
            "source_dtype": str(value.dtype)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.checkpoint / "config.json", args.output / "config.json")
    files = [args.checkpoint / "embedding.pt", args.checkpoint / "decoder.pt"]
    files += sorted(args.checkpoint.glob("block_*.pt"))
    manifest = {"format": "symmetric_int8_per_tensor_v1", "files": []}
    for source in files:
        state = torch.load(source, map_location="cpu", weights_only=True)
        converted = {}
        for key, value in state.items():
            converted[key] = quantize_tensor(value) if torch.is_tensor(value) else value
        target = args.output / source.name
        torch.save(converted, target)
        manifest["files"].append({"name": target.name, "bytes": target.stat().st_size})
        print(target, flush=True)
    (args.output / "quantization_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

