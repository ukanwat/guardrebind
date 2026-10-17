"""Crash-safe file IO for long experiment runs.

A crash or power loss can leave a JSONL file with a torn last line, or a JSON
result half written. Readers here drop a torn tail (and truncate it, so the
next append starts on a clean line); writers go through a temp file and an
atomic rename, so a result file on disk is always complete.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    """Rows of a JSONL file that resumable runs append to. A torn final line is
    cut off the file; any other unparsable line is skipped and reported."""
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        keep = data.rfind(b"\n") + 1
        with path.open("r+b") as f:
            f.truncate(keep)
        print(f"{path}: dropped a torn last line ({len(data) - keep} bytes)", flush=True)
        data = data[:keep]
    rows, bad = [], 0
    for line in data.decode("utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    if bad:
        print(f"{path}: skipped {bad} unparsable lines", flush=True)
    return rows


def atomic_write_text(path: Path, text: str) -> None:
    """Write text so the file is either the old version or the complete new one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
