"""Integrate retrieved Choice evidence with the existing cost and budget pipeline."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from methods.base import BaseRouter
from methods.EquiRouter import resolve_dataset_config
from utils.build_model import init_model
from utils.cost import fit_cost_model

from .backends import ChoiceBackend, ClefFlashBackend, JevBackend
from .cache import ChoiceCache
from .retrieval import TrainingRetriever
from .schema import build_choice_request


def _image_keys(frame) -> list[str]:
    if "image_path" not in frame:
        return [""] * len(frame)
    return [hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in frame["image_path"]]


class SystemOneRouter(BaseRouter):
    """Training-free utility decisions plus the shared supervised cost predictor.

    ``train()`` builds training-only retrieval context and fits costs; it never
    fine-tunes the System One model. ``predict()`` returns the same pair of
    candidate-aligned matrices as EquiRouter and evaluates each query once.
    """

    def __init__(self, args: dict, backend: ChoiceBackend | None = None) -> None:
        super().__init__(args)
        self.config = self.args.get("system_one", {})
        dataset = self.args["dataset"]["name"]
        cost_args = dict(self.args)
        cost_args["model"] = resolve_dataset_config(self.args["model"], dataset)
        self.model_cost = init_model(
            cost_args,
            input_dim=int(self.args["embeddings"]["out_dim"]),
            out_dim=len(self.model_list),
        ).to(self.device)
        self.backend = backend
        self.cache = ChoiceCache(self.config.get("cache_dir"))
        self.retriever = None

    def _backend(self) -> ChoiceBackend:
        if self.backend is None:
            name = self.config.get("backend", "jev")
            if name == "jev":
                self.backend = JevBackend(**self.config.get("jev", {}))
            elif name == "clef-flash":
                settings = dict(self.config.get("clef-flash", {}))
                settings.setdefault("device", str(self.device))
                self.backend = ClefFlashBackend(**settings)
            else:
                raise ValueError(f"Unknown System One backend: {name}")
        return self.backend

    def train(self) -> None:
        embeddings, utility, costs = self._prepare_training_data()
        if not np.isfinite(utility).all():
            raise ValueError("Training utilities must be finite")
        # Keep only inputs, never target labels or response text, as request context.
        columns = ["prompt"] + (["image_path"] if "image" in self.args["modality"] else [])
        self.training_inputs = self.train_df[columns].reset_index(drop=True).copy()
        self.training_utilities = utility.copy()
        retriever = TrainingRetriever(
            embeddings,
            self.training_inputs["prompt"].astype(str).tolist(),
            _image_keys(self.training_inputs),
            k=int(self.config.get("neighbors", 32)),
        )
        training = resolve_dataset_config(self.args["training"], self.args["dataset"]["name"])
        self.cost_mean, self.cost_std = fit_cost_model(
            self.model_cost, embeddings, costs, training, self.device
        )
        self.retriever = retriever

    def predict(self, test_embedding, batch_size: int = 512, *, queries=None):
        """Score test inputs, or explicit input rows supplied via ``queries``.

        Input rows need ``prompt`` and, for multimodal routing, ``image_path``.
        Extra columns (including labels and costs) are deliberately discarded.
        """
        if self.retriever is None:
            raise RuntimeError("Call train() before predict()")
        if isinstance(test_embedding, torch.Tensor):
            test_embedding = test_embedding.detach().cpu().numpy()
        embeddings = np.asarray(test_embedding, dtype=np.float32)
        dimension = int(self.args["embeddings"]["out_dim"])
        if embeddings.ndim != 2 or embeddings.shape[1] != dimension:
            raise ValueError(f"Expected embeddings of shape (N, {dimension})")
        if not np.isfinite(embeddings).all() or batch_size <= 0:
            raise ValueError("Embeddings must be finite and batch_size positive")
        frame = self.test_df if queries is None else queries
        if len(frame) != len(embeddings):
            raise ValueError(
                "Input rows must align with embeddings; provide queries for custom inputs"
            )
        columns = list(self.training_inputs.columns)
        inputs = frame[columns].reset_index(drop=True)
        image_keys = _image_keys(inputs)
        scores = np.empty((len(embeddings), len(self.model_list)), dtype=np.float64)
        if not len(embeddings):
            return scores, scores.copy()
        backend = self._backend()
        for row, vector in enumerate(embeddings):
            target = inputs.iloc[row]
            neighbors = self.retriever.retrieve(vector, str(target["prompt"]), image_keys[row])
            history = self.training_inputs.iloc[neighbors]
            request = build_choice_request(
                backend.model,
                self.model_list,
                str(target["prompt"]),
                history["prompt"].astype(str).tolist(),
                self.training_utilities[neighbors],
                target_image=target["image_path"] if "image_path" in inputs else None,
                history_images=history["image_path"].tolist() if "image_path" in history else None,
            )
            scores[row] = self.cache.scores(backend, request)
        parts = []
        self.model_cost.eval()
        with torch.inference_mode():
            for start in range(0, len(embeddings), batch_size):
                batch = torch.from_numpy(embeddings[start : start + batch_size]).to(self.device)
                cost = self.model_cost(batch) * self.cost_std + self.cost_mean
                parts.append(cost.clamp_min(0).cpu().numpy())
        return scores, self._clip_predicted_costs(np.concatenate(parts))
