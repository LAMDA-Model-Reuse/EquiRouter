"""Embedding backends required by the four EquiRouter benchmarks."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.sources import MINILM_REVISION, file_sha256


class Embedder:
    """Create MiniLM or CLIP features and concatenate active modalities."""

    def __init__(self, args: dict) -> None:
        self.config = args["embeddings"]
        if self.config.get("training", False):
            raise ValueError("The released EquiRouter uses frozen embedding encoders")
        self.normalize = bool(self.config.get("normalize", False))
        self.out_dim = int(self.config["out_dim"])
        self.text_embedder = None
        self.image_embedder = None
        text_model = self.config.get("text_model")
        image_model = self.config.get("image_model")
        self.cache_identity = {"text_model_revision": MINILM_REVISION}
        local_checkpoint = self.config.get("text_model_dir")
        if local_checkpoint:
            root = Path(local_checkpoint)
            self.cache_identity["local_checkpoint"] = {
                str(path.relative_to(root)): file_sha256(path)
                for path in sorted(root.rglob("*"))
                if path.is_file()
                and not any(part.startswith(".") for part in path.relative_to(root).parts)
            }
        clip_runtime = None
        if text_model == "ViT-B/16" or image_model == "ViT-B/16":
            import clip
            from clip.clip import _MODELS

            clip_runtime = clip.load("ViT-B/16", device=_clip_device(self.config))
            clip_runtime[0].eval().requires_grad_(False)
            self.cache_identity["clip_checkpoint"] = _MODELS["ViT-B/16"].split("/")[-2]
        if text_model == "all-MiniLM-L6-v2":
            self.text_embedder = _create_minilm(self.config)
        elif text_model == "ViT-B/16":
            self.text_embedder = _create_clip_text(self.config, clip_runtime)
        elif text_model is not None:
            raise ValueError(f"Unsupported EquiRouter text encoder: {text_model}")
        if image_model == "ViT-B/16":
            self.image_embedder = _create_clip_image(self.config, clip_runtime)
        elif image_model is not None:
            raise ValueError(f"Unsupported EquiRouter image encoder: {image_model}")

    def run_embed(self, texts=None, images=None) -> torch.Tensor:
        batch_size = int(self.config.get("batch_size", 256))
        parts: list[torch.Tensor] = []
        row_count: int | None = None
        for values, encoder, label in (
            (texts, self.text_embedder, "Text"),
            (images, self.image_embedder, "Image"),
        ):
            if values is None:
                continue
            if encoder is None:
                raise ValueError(f"{label.lower()} input has no configured encoder")
            items = [values] if isinstance(values, (str, os.PathLike)) else list(values)
            batches = [
                encoder(items[start : start + batch_size]).float()
                for start in tqdm(range(0, len(items), batch_size), desc=f"{label} Embedding")
            ]
            if not batches:
                raise ValueError("Cannot embed an empty input sequence")
            features = torch.cat(batches, dim=0)
            if row_count is not None and row_count != len(features):
                raise ValueError("Text and image input counts must match")
            row_count = len(features)
            parts.append(features)
        if not parts:
            raise ValueError("No inputs were provided to the embedder")
        combined = torch.cat(parts, dim=1)
        if combined.shape[1] != self.out_dim:
            raise ValueError(
                f"Configured embedding dimension is {self.out_dim}, got {combined.shape[1]}"
            )
        if self.normalize:
            combined = combined / combined.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return combined


def _create_minilm(config: dict):
    from sentence_transformers import SentenceTransformer

    configured_path = config.get("text_model_dir")
    if configured_path is not None and not Path(configured_path).is_dir():
        raise FileNotFoundError(f"MiniLM checkpoint not found at {configured_path}")
    model_source = configured_path or "sentence-transformers/all-MiniLM-L6-v2"
    device = config.get("device", "cpu")
    model = SentenceTransformer(
        str(model_source), device=device, revision=None if configured_path else MINILM_REVISION
    )
    model.eval().requires_grad_(False)
    logging.info("Loaded MiniLM from %s", model_source)

    @torch.no_grad()
    def encode(texts: Sequence[str]) -> torch.Tensor:
        values = model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False,
        )
        return torch.from_numpy(values).float().to(device)

    return encode


def _clip_device(config: dict) -> torch.device:
    requested = torch.device(config.get("device", "cpu"))
    return (
        requested if requested.type != "cuda" or torch.cuda.is_available() else torch.device("cpu")
    )


def _create_clip_text(config: dict, runtime):
    import clip

    device = _clip_device(config)
    model, _ = runtime
    use_fp16 = bool(config.get("use_fp16", False)) and device.type == "cuda"

    @torch.no_grad()
    def encode(texts: Sequence[str]) -> torch.Tensor:
        tokens = clip.tokenize(list(texts), truncate=True).to(device)
        with torch.autocast(device_type="cuda", enabled=use_fp16):
            features = model.encode_text(tokens).float()
        return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    return encode


class _ImageDataset(Dataset):
    def __init__(self, items: Sequence, preprocess) -> None:
        self.items = list(items)
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> torch.Tensor:
        item = self.items[index]
        try:
            if isinstance(item, (str, os.PathLike)):
                image = Image.open(item).convert("RGB")
            elif isinstance(item, Image.Image):
                image = item.convert("RGB")
            else:
                array = np.asarray(item)
                if array.ndim == 2:
                    array = np.stack([array] * 3, axis=-1)
                image = Image.fromarray(array[..., :3].astype("uint8"), "RGB")
            return self.preprocess(image)
        except (OSError, ValueError, TypeError) as error:
            raise ValueError(f"Could not decode image at index {index}") from error


def _create_clip_image(config: dict, runtime):
    device = _clip_device(config)
    model, preprocess = runtime
    use_fp16 = bool(config.get("use_fp16", False)) and device.type == "cuda"

    @torch.no_grad()
    def encode(images: Sequence) -> torch.Tensor:
        loader = DataLoader(
            _ImageDataset(images, preprocess),
            batch_size=int(config.get("batch_size", 256)),
            shuffle=False,
            num_workers=int(config.get("image_workers", 8)),
            pin_memory=device.type == "cuda",
        )
        parts: list[torch.Tensor] = []
        for batch in loader:
            batch = batch.to(device).float()
            with torch.autocast(device_type="cuda", enabled=use_fp16):
                features = model.encode_image(batch).float()
            parts.append(features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12))
        return torch.cat(parts, dim=0)

    return encode
