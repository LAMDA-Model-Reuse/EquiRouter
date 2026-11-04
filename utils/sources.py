"""Pinned upstream data identities and verification before deserialization."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

from huggingface_hub import hf_hub_download

DATA_SOURCES = {
    "withmartian/routerbench": {
        "revision": "784021482c3f320c6619ed4b3bb3b41a21424fcb",
        "sha256": {
            "routerbench_raw.pkl": "18e919267500a1d49302e0efe25cf7d9aa2f5b74aa209f5230442c2cfbee7971",
        },
    },
    "gh0stHunter/MMR-Bench": {
        "revision": "b2619d8389935171022f0a7e8bee45f80ca249dd",
        "sha256": {
            "MMR-Bench.csv": "a7bfe31d38269361bcd778b9afb8209e62824da0f77d4f696f972b6020d07ae5",
            "images.tar.gz": "32d1fb2e41165e5a7c5762457cf08cf7536536d9968c895053ac06d9d14b861e",
        },
    },
    "llm-blender/mix-instruct": {
        "revision": "c7f77a4ef0515a99d1752c6387c22366d6922da6",
        "sha256": {},
    },
    "linggm/Routereval": {
        "revision": "0ee4de2661854b588474c211bebe97dc1e434237",
        "sha256": {
            "router_dataset.zip": "cb17dda3ca346a2d4eb429b5afd33359c343e2b4bb33ed14611211332ad802e7",
            "arc_router_dataset.pkl": "06a729536ec0373086c2520d96a5433ae42568b53208cf5105594312968aa270",
            "gsm8k_router_dataset.pkl": "306af051c4af6c0cd4ef45393e36d5dcea15312a8cbb7d4c7ee1e6bb61e2f8ac",
            "harness_truthfulqa_mc_0_router_dataset.pkl": "de7cd569aae53537f328bc5c6872602312ef31ac63201798a0577f9a00ce6441",
            "hellaswag_router_dataset.pkl": "386339921ec6e2694a5319bc124f9fe672af522e2df8742a3c8524f9060dd855",
            "mmlu_router_dataset.pkl": "9e0f242f648cc943b1cdb067199894142d21cd35121f1ffdb0e18005d3077f45",
            "winogrande_router_dataset.pkl": "009f354ccfa8f943a7d11df48fa5d06d15dc81a774eb34aa8412b4cc9c8dcc16",
        },
    },
}

MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path: str | Path, expected_sha256: str) -> None:
    if file_sha256(path) != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {Path(path).name}; refusing to use this asset")


def download_asset(repo_id: str, filename: str, directory: str, source: dict) -> Path:
    """Download one fixed revision and verify even an already-present local file."""
    destination = Path(directory) / filename
    if not destination.is_file():
        destination = Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=source["revision"],
                filename=filename,
                local_dir=directory,
            )
        )
    verify_file(destination, source["sha256"][filename])
    return destination


def load_verified_pickle(path: str | Path, expected_sha256: str):
    """Load only a byte-verified, trusted upstream pickle through the same handle."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
        if digest.hexdigest() != expected_sha256:
            raise ValueError(f"SHA-256 mismatch for {Path(path).name}; pickle was not loaded")
        handle.seek(0)
        return pickle.load(handle)
