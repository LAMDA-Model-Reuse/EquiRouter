"""Command-line interface for EquiRouter and System One routing baselines."""

import argparse
import json
from pathlib import Path
from typing import Any

from train import train

PROJECT_ROOT = Path(__file__).resolve().parent


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_benchmark_config(requested_name: str) -> Path:
    directory = PROJECT_ROOT / "configs/benchmarks"
    matches = [
        path
        for path in directory.glob("*.json")
        if path.stem.casefold() == requested_name.casefold()
    ]
    if not matches:
        available = ", ".join(sorted(path.stem for path in directory.glob("*.json")))
        raise FileNotFoundError(f"Unknown benchmark {requested_name!r}. Available: {available}")
    return matches[0]


def load_config(
    dataset: str, device: str | None = None, method: str = "EquiRouter"
) -> dict[str, Any]:
    """Load benchmark and method settings independently of the working directory."""
    config = _load_json(_resolve_benchmark_config(dataset))
    config.update(_load_json(PROJECT_ROOT / "configs/routers/EquiRouter.json"))
    methods = {"equirouter": "EquiRouter", "jev": "Jev", "clef-flash": "ClefFlash"}
    if method.casefold() not in methods:
        raise ValueError(f"Unknown method {method!r}. Available: {', '.join(methods)}")
    config["method"] = methods[method.casefold()]
    if config["method"] != "EquiRouter":
        config["system_one"] = _load_json(PROJECT_ROOT / "configs/system_one/default.json")
        backend = "jev" if config["method"] == "Jev" else "clef-flash"
        config["system_one"]["backend"] = backend
    if device is not None:
        config["device"] = device
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Benchmark name.")
    parser.add_argument(
        "--method", choices=("equirouter", "jev", "clef-flash"), default="equirouter"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Execution device override: auto, cpu, cuda, or cuda:N.",
    )
    arguments = parser.parse_args()
    train(load_config(arguments.dataset, device=arguments.device, method=arguments.method))


if __name__ == "__main__":
    main()
