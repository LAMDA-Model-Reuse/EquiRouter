"""Public run metadata without host names, credentials, or benchmark contents."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
from pathlib import Path

import torch

from utils.sources import file_sha256


def source_hashes(root: Path) -> dict[str, str]:
    paths = [
        root / name for name in ("main.py", "train.py", "requirements.txt", "requirements-clef.txt")
    ]
    for directory, pattern in (
        ("methods", "*.py"),
        ("utils", "*.py"),
        ("scripts", "*.py"),
        ("configs", "*.json"),
    ):
        paths.extend((root / directory).rglob(pattern))
    return {str(path.relative_to(root)): file_sha256(path) for path in sorted(paths)}


def split_identity(frame) -> dict:
    columns = [name for name in ("sample_id", "prompt") if name in frame]
    digest = hashlib.sha256()
    values = (
        frame[columns].astype(str).itertuples(index=False, name=None)
        if columns
        else ((str(i),) for i in range(len(frame)))
    )
    for row in values:
        for value in row:
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return {"rows": len(frame), "ordered_inputs_sha256": digest.hexdigest()}


def environment_info(device: str) -> dict:
    packages = {
        dist.metadata["Name"]: dist.version
        for dist in importlib.metadata.distributions()
        if dist.metadata["Name"]
    }
    return {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "device": device,
        "torch_threads": torch.get_num_threads(),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(torch.device(device))
        if str(device).startswith("cuda")
        else None,
        "packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
    }
