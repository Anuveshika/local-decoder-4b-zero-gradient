"""Create SHA-256 manifest for repository release artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "RELEASE_MANIFEST.json"
records = []
for path in sorted(ROOT.rglob("*")):
    if not path.is_file() or path == TARGET or ".git" in path.parts or ".mplconfig" in path.parts:
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    records.append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
                    "sha256": digest})
TARGET.write_text(json.dumps({"schema_version": 1, "files": records}, indent=2), encoding="utf-8")
print(TARGET, len(records))
