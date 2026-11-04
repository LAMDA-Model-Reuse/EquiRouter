"""Cache validated scores by request content and backend version."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .schema import parse_choice


class ChoiceCache:
    """Keep requests and credentials out of stored cache records."""

    def __init__(self, directory: str | Path | None = None) -> None:
        self.directory = Path(directory) if directory is not None else None
        self.memory: dict[str, np.ndarray] = {}

    def scores(self, backend, request: dict) -> np.ndarray:
        serialized = json.dumps(
            {"schema_version": 1, "backend": backend.cache_identity, "request": request},
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        key = hashlib.sha256(serialized.encode()).hexdigest()
        if key in self.memory:
            return self.memory[key].copy()
        options = list(request["questions"]["selected_model"]["criteria"])
        path = self.directory / f"{key}.json" if self.directory is not None else None
        if path is not None and path.is_file():
            response = json.loads(path.read_text(encoding="utf-8"))
            values = parse_choice(response, options)
        else:
            values = parse_choice(backend.predict(request), options)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                response = {
                    "answers": {
                        "selected_model": {
                            "choice": options[int(values.argmax())],
                            "probabilities": dict(zip(options, values.tolist())),
                        }
                    }
                }
                temporary = path.with_name(f".{key}.{os.getpid()}.tmp")
                temporary.write_text(json.dumps(response, allow_nan=False), encoding="utf-8")
                os.replace(temporary, path)
        self.memory[key] = values.copy()
        return values.copy()
