"""Deterministic cosine retrieval from training inputs only."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def normalize_query(text: str) -> str:
    return " ".join(str(text).split()).casefold()


class TrainingRetriever:
    """Retrieve distinct training inputs, excluding target text/image duplicates."""

    def __init__(
        self,
        embeddings,
        queries: Sequence[str],
        image_keys: Sequence[str] | None = None,
        k: int = 32,
    ) -> None:
        values = np.asarray(embeddings, dtype=np.float32)
        if values.ndim != 2 or len(values) != len(queries) or not len(values):
            raise ValueError("Training embeddings must align with non-empty queries")
        if k <= 0 or not np.isfinite(values).all():
            raise ValueError("k must be positive and embeddings finite")
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if (norms == 0).any():
            raise ValueError("Cosine retrieval requires non-zero training embeddings")
        self.embeddings = values / norms
        self.queries = [normalize_query(q) for q in queries]
        self.image_keys = list(image_keys) if image_keys is not None else [""] * len(values)
        if len(self.image_keys) != len(values):
            raise ValueError("Training image keys must align with queries")
        self.k = int(k)

    def retrieve(self, embedding, query: str, image_key: str = "") -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32)
        if vector.shape != (self.embeddings.shape[1],) or not np.isfinite(vector).all():
            raise ValueError("Target embedding must be finite and match the training dimension")
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("Cosine retrieval requires a non-zero target embedding")
        similarity = self.embeddings @ (vector / norm)
        order = np.argsort(-similarity, kind="stable")
        seen_queries = {normalize_query(query)}
        seen_images = {image_key} if image_key else set()
        selected = []
        for index in order:
            text, image = self.queries[index], self.image_keys[index]
            if text in seen_queries or (image and image in seen_images):
                continue
            selected.append(int(index))
            seen_queries.add(text)
            if image:
                seen_images.add(image)
            if len(selected) == self.k:
                return np.asarray(selected, dtype=np.int64)
        raise ValueError(
            f"Need {self.k} distinct training inputs after duplicate exclusion; found {len(selected)}"
        )
