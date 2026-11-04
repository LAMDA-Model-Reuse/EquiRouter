"""Jev HTTP and Clef-Flash native Choice backends.

No credentials or checkpoints are read until inference is explicitly requested.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import importlib.util
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import torch
from PIL import Image


class ChoiceBackend(Protocol):
    model: str

    @property
    def cache_identity(self) -> dict: ...

    def predict(self, request: dict) -> dict: ...


def _https_origin(url: str) -> tuple[str, int]:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ValueError("System One URLs must use HTTPS without credentials or fragments")
    return parsed.hostname.casefold(), parsed.port or 443


class _AuthenticatedRedirectHandler(HTTPRedirectHandler):
    """Reject insecure redirects and cross-origin forwarding of credentials."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        try:
            same_origin = _https_origin(req.full_url) == _https_origin(newurl)
        except ValueError:
            raise HTTPError(req.full_url, code, "Unsafe redirect blocked", headers, fp) from None
        authenticated = any(name.casefold() == "authorization" for name, _ in req.header_items())
        if authenticated and not same_origin:
            raise HTTPError(req.full_url, code, "Cross-origin redirect blocked", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _https_urlopen(request: Request, *, timeout: float):
    _https_origin(request.full_url)
    return build_opener(_AuthenticatedRedirectHandler()).open(request, timeout=timeout)


class JevBackend:
    """TypeSafe System One client with bounded retries and environment credentials."""

    def __init__(
        self,
        model: str = "jev-1.14.0",
        endpoint: str | None = None,
        api_key_env: str = "TYPESAFE_API_KEY",
        timeout: float = 120,
        retries: int = 3,
    ) -> None:
        self.model = model
        self.endpoint = endpoint or os.environ.get(
            "TYPESAFE_ENDPOINT", "https://api.typesafe.ai/v1/systemone"
        )
        _https_origin(self.endpoint)
        if timeout <= 0 or retries < 0:
            raise ValueError("timeout must be positive and retries non-negative")
        self.api_key_env = api_key_env
        self.timeout = float(timeout)
        self.retries = int(retries)

    @property
    def cache_identity(self) -> dict:
        return {
            "backend": "typesafe-systemone-assets-v1",
            "model": self.model,
            "endpoint_hash": hashlib.sha256(self.endpoint.encode()).hexdigest(),
        }

    @staticmethod
    def _wire_request(request: dict) -> dict:
        result = copy.deepcopy(request)
        images = result.pop("images", [])
        if not images:
            return result
        references = []
        assets = {}
        for index, data_url in enumerate(images):
            header, encoded = data_url.split(",", 1)
            image_format = "png" if header == "data:image/png;base64" else None
            if image_format is None:
                raise ValueError("Expected canonical PNG data URLs")
            token = hashlib.sha256(f"{index}:{data_url}".encode()).hexdigest()
            reference = f"__equirouter_image_{token}__"
            references.append(reference)
            assets[reference] = {"type": "image", "format": image_format, "data_base64": encoded}

        def replace(value):
            if isinstance(value, dict):
                if set(value) == {"image_index"}:
                    return references[value["image_index"]]
                return {key: replace(item) for key, item in value.items()}
            if isinstance(value, list):
                return [replace(item) for item in value]
            return value

        result["state"].pop("image_index_definition", None)
        result["state"] = replace(result["state"])
        result["assets"] = assets
        return result

    def predict(self, request: dict) -> dict:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(f"Set {self.api_key_env} before using the Jev backend")
        wire_request = self._wire_request(request)
        body = json.dumps(wire_request, ensure_ascii=True, allow_nan=False).encode()
        if len(body) > 32 * 1024 * 1024:
            raise ValueError("System One request exceeds 32 MiB; evidence was not truncated")
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        if "assets" in wire_request:
            headers["TypeSafe-Danger-Zone-Images"] = "true"
        http_request = Request(self.endpoint, data=body, headers=headers, method="POST")
        for attempt in range(self.retries + 1):
            try:
                with _https_urlopen(http_request, timeout=self.timeout) as response:
                    return json.load(response)
            except HTTPError as error:
                code = error.code
                error.close()
                if code not in {408, 429, 500, 502, 503, 504, 529} or attempt == self.retries:
                    raise RuntimeError(f"System One request failed with HTTP {code}") from None
            except (URLError, TimeoutError):
                if attempt == self.retries:
                    raise RuntimeError("System One request failed after bounded retries") from None
            time.sleep(min(2**attempt, 10))
        raise RuntimeError("System One request did not complete")


class ClefFlashBackend:
    """Run the publisher's pinned schema head locally without free-form generation."""

    MODEL_ID = "Cloudflare/clef-flash"
    REVISION = "f0b7d3cd65296abfe459e6c228b618f5d8b26d2f"

    def __init__(
        self,
        model_dir: str | Path | None = None,
        device: str = "cuda",
        revision: str = REVISION,
        max_length: int | None = None,
    ) -> None:
        self.model = self.MODEL_ID
        configured = model_dir or os.environ.get("CLEF_MODEL_DIR")
        self.model_dir = Path(configured) if configured is not None else None
        self._checkpoint_origin = (
            str(self.model_dir.resolve()) if self.model_dir is not None else "hub"
        )
        self.device = torch.device(device)
        self.revision = revision
        if max_length is not None and max_length <= 0:
            raise ValueError("max_length must be positive")
        self.max_length = max_length
        self._runtime = None

    @property
    def cache_identity(self) -> dict:
        # A local override is a separate cache namespace. Users must clear its
        # cache if replacing weights in place with a different checkpoint.
        local = self._checkpoint_origin
        return {
            "backend": "clef-native-v1",
            "model": self.model,
            "revision": self.revision,
            "checkpoint_hash": hashlib.sha256(local.encode()).hexdigest(),
            "max_length": self.max_length,
        }

    def _load(self) -> None:
        if self._runtime is not None:
            return
        if self.model_dir is None:
            from huggingface_hub import snapshot_download

            self.model_dir = Path(snapshot_download(self.model, revision=self.revision))
        module_path = self.model_dir / "joint_schema_model.py"
        if not module_path.is_file():
            raise FileNotFoundError("Clef checkpoint is missing joint_schema_model.py")
        module_name = (
            "_equirouter_clef_" + hashlib.sha256(str(module_path).encode()).hexdigest()[:12]
        )
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        dtype = torch.float32 if self.device.type == "cpu" else torch.bfloat16
        self._network, self._processor = module.load_release_model(
            self.model_dir,
            device=self.device,
            dtype=dtype,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        self._network.requires_grad_(False)
        config = json.loads((self.model_dir / "config.json").read_text())
        backbone_limit = int(config["text_config"]["max_position_embeddings"])
        self._context_limit = min(self.max_length or backbone_limit, backbone_limit)
        self._runtime = module

    def predict(self, request: dict) -> dict:
        self._load()
        record = copy.deepcopy(request)
        if "images" in record:
            decoded = []
            for data_url in record["images"]:
                with Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))) as image:
                    decoded.append(image.convert("RGB"))
            record["images"] = decoded
        # The publisher truncates state by default. Encode the complete record
        # first and reject over-limit evidence explicitly instead.
        encoded = self._runtime.encode_record(
            self._processor.tokenizer,
            record,
            max_length=2**31 - 1,
            processor=self._processor,
        )
        if len(encoded.input_ids) > self._context_limit:
            raise ValueError(
                f"Complete evidence needs {len(encoded.input_ids)} tokens, exceeding "
                f"the context limit {self._context_limit}; no truncation was applied"
            )
        batch = self._runtime.collate_records(
            [encoded], self._processor.tokenizer.pad_token_id, self.device
        )
        with torch.inference_mode():
            logits = self._network(batch)[0]
        answers = {}
        for question, values in zip(encoded.questions, logits):
            probabilities = values.float().softmax(-1).tolist()
            options = dict(zip(question.option_ids, probabilities))
            answers[question.question_id] = {
                "choice": max(options, key=options.__getitem__),
                "probabilities": options,
            }
        return {"model": self.model, "answers": answers}
