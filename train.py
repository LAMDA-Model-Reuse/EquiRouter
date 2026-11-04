"""Shared training and evaluation entry point for routing methods."""

import logging
import time
from pathlib import Path
from typing import Any

from methods import EquiRouter


def setup_logging(args: dict[str, Any]) -> Path:
    """Configure console and file logging for one run."""
    dataset_name = args.get("dataset", {}).get("name", "unknown_dataset")
    log_dir = Path("logs") / dataset_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{args.get('method', 'EquiRouter')}_{time.time_ns()}.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(file_handler)
    root.addHandler(console_handler)
    logging.info("Logging to %s", log_path)
    return log_path


def train(args: dict[str, Any]) -> None:
    """Fit a router's training components and evaluate the held-out partition."""
    setup_logging(args)
    method = args.get("method", "EquiRouter")
    if method == "EquiRouter":
        router = EquiRouter(args)
    elif method in {"Jev", "ClefFlash"}:
        from methods.system_one import SystemOneRouter

        router = SystemOneRouter(args)
    else:
        raise ValueError(f"Unknown router method: {method}")
    logging.info("Preparing %s...", method)
    router.train()
    logging.info("Evaluating %s...", method)
    router.evaluate()
