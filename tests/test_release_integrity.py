"""Regression coverage for dataset identity, integrity, splitting, and HTTPS."""

import hashlib
import io
import zipfile
from email.message import Message
from unittest.mock import Mock
from urllib.error import HTTPError
from urllib.request import HTTPSHandler, Request, build_opener
from urllib.response import addinfourl

import numpy as np
import pandas as pd
import pytest
import torch

from methods.system_one.backends import JevBackend, _AuthenticatedRedirectHandler
from utils.cost import make_optimizer
from utils.data import BaseDatasetLoader, MMRBenchLoader, RouterEvalLoader
from utils.sources import download_asset, load_verified_pickle


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://example.com",
        "http://localhost",
        "https://u:p@example.com",
        "https://example.com/#private",
    ],
)
def test_jev_rejects_insecure_endpoints(endpoint):
    with pytest.raises(ValueError, match="HTTPS"):
        JevBackend(endpoint=endpoint)


@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
@pytest.mark.parametrize(
    "target",
    [
        "https://other.example/path",
        "https://api.example:444/path",
        "http://api.example/path",
        "https://api.example@other.example/path",
    ],
)
def test_authenticated_redirects_cannot_change_origin_or_downgrade(code, target):
    request = Request("https://api.example/start", headers={"Authorization": "Bearer dummy"})
    with pytest.raises(HTTPError):
        _AuthenticatedRedirectHandler().redirect_request(
            request, None, code, "redirect", {}, target
        )


def test_same_origin_https_redirect_is_allowed():
    request = Request("https://API.example/start", headers={"Authorization": "Bearer dummy"})
    redirected = _AuthenticatedRedirectHandler().redirect_request(
        request, None, 302, "redirect", {}, "https://api.example:443/next"
    )
    assert redirected.get_header("Authorization") == "Bearer dummy"


def test_urllib_chain_blocks_redirect_before_second_request():
    visited = []

    class FakeTLS(HTTPSHandler):
        def https_open(self, request):
            visited.append(request.full_url)
            headers = Message()
            headers["Location"] = "https://other.example/receive"
            response = addinfourl(io.BytesIO(b""), headers, request.full_url, 302)
            response.msg = "redirect"
            return response

    opener = build_opener(FakeTLS(), _AuthenticatedRedirectHandler())
    request = Request("https://api.example/start", headers={"Authorization": "Bearer dummy"})
    with pytest.raises(HTTPError):
        opener.open(request)
    assert visited == ["https://api.example/start"]


@pytest.mark.parametrize(
    "ratios",
    [
        {"train": 0.4},
        {"train": 0.4, "test": 0.1},
        {"train": 0.4, "test": 0.6, "val": 0},
        {"train": np.nan, "test": 0.6},
        {"train": 1, "test": 0},
    ],
)
@pytest.mark.parametrize("mode", ["in-domain", "out-of-domain"])
def test_invalid_split_configuration_is_rejected(ratios, mode):
    loader = BaseDatasetLoader(
        None, {"seed": 42, "dataset": {"split": {"mode": mode, "ratios": ratios}}}
    )
    with pytest.raises(ValueError):
        loader.split(pd.DataFrame({"id": range(10), "eval_name": list("aabbccddee")}))


def test_split_uses_declared_ratios_without_dropping_rows():
    args = {
        "seed": 42,
        "dataset": {"split": {"mode": "in-domain", "ratios": {"train": 0.4, "test": 0.6}}},
    }
    loader = BaseDatasetLoader(None, args)
    train, test = loader.split(pd.DataFrame({"id": range(101)}))
    assert (len(train), len(test)) == (40, 61)
    assert set(train.id).isdisjoint(test.id)
    assert set(train.id) | set(test.id) == set(range(101))


def test_tampered_pickle_is_rejected_before_deserialization(tmp_path, monkeypatch):
    path = tmp_path / "data.pkl"
    path.write_bytes(b"untrusted bytes")
    deserialize = Mock()
    monkeypatch.setattr("utils.sources.pickle.load", deserialize)
    with pytest.raises(ValueError, match="SHA-256"):
        load_verified_pickle(path, "0" * 64)
    deserialize.assert_not_called()


def test_cached_upstream_asset_is_also_verified(tmp_path, monkeypatch):
    path = tmp_path / "data.pkl"
    path.write_bytes(b"modified")
    download = Mock()
    monkeypatch.setattr("utils.sources.hf_hub_download", download)
    source = {"revision": "pinned", "sha256": {path.name: hashlib.sha256(b"original").hexdigest()}}
    with pytest.raises(ValueError, match="SHA-256"):
        download_asset("official/data", path.name, str(tmp_path), source)
    download.assert_not_called()


def test_mmrbench_preserves_missing_nano_values_without_other_model_fallback(tmp_path):
    loader = MMRBenchLoader(str(tmp_path), {})
    names = [
        "Qwen2.5-VL-3B-Instruct",
        "Qwen2.5-VL-72B-Instruct",
        "InternVL3-78B",
        "Gemma3-4B",
        "Qwen2.5-VL-7B-Instruct",
        "Claude3-7V_Sonnet",
        "gpt-5-nano-2025-08-07",
        "GeminiPro2-5",
        "gpt-5-2025-08-07",
        "GeminiFlash2-5",
    ]
    from PIL import Image

    (tmp_path / "MathVision").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "MathVision/0.png")
    row = {"dataset_idx": "MathVision_0", "question": "query", "answer": "a"}
    for name in [*names, "gpt-5-minimal-2025-08-07"]:
        row.update({name + "_correct": True, name + "_cost": 1.0, name + "_prediction": "a"})
    for suffix in ["_correct", "_cost", "_prediction"]:
        row[names[6] + suffix] = np.nan
    pd.DataFrame([row]).to_csv(tmp_path / "MMR-Bench.csv", index=False)
    models, frame = loader.process(str(tmp_path))
    assert models == names
    assert pd.isna(frame.loc[0, "model_6_performance"])
    assert pd.isna(frame.loc[0, "model_6_cost"])
    assert pd.isna(frame.loc[0, "model_6_prediction"])
    assert len(frame) == 1


@pytest.mark.parametrize("revision", [None, "main", "a" * 39])
def test_custom_loader_requires_pinned_metadata_before_network_access(monkeypatch, revision):
    class CustomLoader(BaseDatasetLoader):
        hf_id = "example/custom"
        source_metadata = {"revision": revision}

    download = Mock()
    monkeypatch.setattr("utils.data.load_dataset", download)
    with pytest.raises(ValueError, match="source_metadata"):
        CustomLoader(None, {}).download()
    download.assert_not_called()


def test_custom_loader_uses_subclass_source_metadata(tmp_path, monkeypatch):
    class CustomLoader(BaseDatasetLoader):
        hf_id = "example/custom"
        source_metadata = {"revision": "a" * 40}

    download = Mock()
    monkeypatch.setattr("utils.data.load_dataset", download)
    args = {"dataset": {}}
    loader = CustomLoader(str(tmp_path), args)
    assert loader.download() is download.return_value
    download.assert_called_once_with("example/custom", revision="a" * 40, cache_dir=str(tmp_path))
    loader.source["revision"] = "b" * 40
    assert args["dataset"]["source"]["revision"] == "a" * 40
    assert CustomLoader.source_metadata["revision"] == "a" * 40


@pytest.fixture
def routereval_assets(tmp_path):
    contents = {"arc_router_dataset.pkl": b"arc", "gsm8k_router_dataset.pkl": b"gsm8k"}
    archive_path = tmp_path / "router_dataset.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name, payload in contents.items():
            archive.writestr(f"router_dataset/{name}", payload)
        archive.writestr("router_dataset/unused_task.pkl", b"unused")
    loader = RouterEvalLoader(str(tmp_path), {})
    loader.source["sha256"] = {
        name: hashlib.sha256(payload).hexdigest() for name, payload in contents.items()
    }
    loader.source["sha256"][archive_path.name] = hashlib.sha256(
        archive_path.read_bytes()
    ).hexdigest()
    return loader, contents


def test_routereval_restores_missing_files_and_reuses_verified_files(tmp_path, routereval_assets):
    loader, contents = routereval_assets
    existing = tmp_path / "arc_router_dataset.pkl"
    existing.write_bytes(contents[existing.name])
    original_stat = existing.stat()
    for _ in range(2):
        assert loader.download() == str(tmp_path)
        assert all((tmp_path / name).read_bytes() == payload for name, payload in contents.items())
        assert existing.stat().st_ino == original_stat.st_ino
        assert existing.stat().st_mtime_ns == original_stat.st_mtime_ns
        assert not list(tmp_path.glob(".routereval-*"))
        assert not (tmp_path / "unused_task.pkl").exists()


@pytest.mark.parametrize("complete", [False, True])
def test_routereval_rejects_corrupt_existing_files(tmp_path, routereval_assets, complete):
    loader, contents = routereval_assets
    if complete:
        for name, payload in contents.items():
            (tmp_path / name).write_bytes(payload)
    corrupted = tmp_path / "arc_router_dataset.pkl"
    corrupted.write_bytes(b"modified")
    with pytest.raises(ValueError, match="SHA-256"):
        loader.download()
    assert corrupted.read_bytes() == b"modified"
    if not complete:
        assert not (tmp_path / "gsm8k_router_dataset.pkl").exists()


def test_routereval_verifies_all_missing_members_before_installing(tmp_path, routereval_assets):
    loader, contents = routereval_assets
    loader.source["sha256"]["gsm8k_router_dataset.pkl"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        loader.download()
    assert not any((tmp_path / name).exists() for name in contents)
    assert not list(tmp_path.glob(".routereval-*"))


@pytest.mark.parametrize("name", ["adam", "adamw", "sgd"])
def test_shared_optimizer_preserves_weight_decay(name):
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = make_optimizer([parameter], {"optimizer": name, "lr": 0.1, "weight_decay": 0.2})
    assert optimizer.param_groups[0]["lr"] == 0.1
    assert optimizer.param_groups[0]["weight_decay"] == 0.2
