"""Evaluate EquiRouter with the provided benchmark configurations."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from main import load_config
from methods.EquiRouter import EquiRouter, resolve_dataset_config
from train import setup_logging
from utils.metrics import evaluate_budget_curve
from utils.provenance import environment_info, source_hashes, split_identity

DATASETS = {
    "routerbench": "Routerbench",
    "mmrbench": "MMRBench",
    "mixinstruct": "Mixinstruct",
    "routereval": "RouterEval",
}


def evaluate_predictions(
    router: EquiRouter,
    frame,
    predicted_performance: np.ndarray,
    predicted_cost: np.ndarray,
) -> dict[str, Any]:
    true_performance, true_cost = router._target_matrices(frame)
    budget_points = int(router.args["paper_reproduction"]["budget_points"])
    return evaluate_budget_curve(
        predicted_performance,
        predicted_cost,
        true_performance,
        true_cost,
        model_names=router.model_list,
        budget_points=budget_points,
    )


def _run_standard(config: dict[str, Any]) -> tuple[EquiRouter, dict[str, Any]]:
    started = time.perf_counter()
    router = EquiRouter(config)
    initialized = time.perf_counter()
    router.train()
    trained = time.perf_counter()
    test = router.test_df.reset_index(drop=True)
    embeddings = router._prepare_embeddings(test, "test")
    performance, cost = router.predict(embeddings)
    cost = router._clip_predicted_costs(cost)
    metrics = evaluate_predictions(router, test, performance, cost)
    router.run_timings = {
        "data_and_encoder_setup_seconds": initialized - started,
        "training_with_embedding_seconds": trained - initialized,
        "test_embedding_prediction_evaluation_seconds": time.perf_counter() - trained,
    }
    return router, metrics


def run(dataset: str, device: str, output_dir: Path) -> dict[str, Any]:
    key = dataset.casefold()
    if key not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    config = load_config(DATASETS[key], device=device)
    reproduction = config["paper_reproduction"]
    config["dataset"]["split"] = {
        "mode": reproduction["split"]["mode"],
        "ratios": {
            "train": float(reproduction["split"]["train"]),
            "test": float(reproduction["split"]["test"]),
        },
    }
    setup_logging(config)
    router, metrics = _run_standard(config)
    result = {
        "dataset": router.args["dataset"]["name"],
        "method": "EquiRouter",
        "seed": router.seed,
        "split": config["dataset"]["split"],
        "configuration": resolve_dataset_config(config["training"], router.args["dataset"]["name"]),
        "full_configuration": router.args,
        "model_order": router.model_list,
        "partitions": {
            "train": split_identity(router.train_df),
            "test": split_identity(router.test_df),
        },
        "environment": environment_info(str(router.args.get("device", device))),
        "source_sha256": source_hashes(Path(__file__).resolve().parents[1]),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "timings": router.run_timings,
        "metrics": metrics,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{key}.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"{result['dataset']}: Peak={metrics['peak']:.6f} "
        f"QNC={metrics['qnc']} nAUC={metrics['normalized_auc']:.6f} -> {output}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=(*DATASETS, "all"),
        default="all",
        help="Benchmark to reproduce (default: all four).",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/equirouter-paper"))
    args = parser.parse_args()
    selected = DATASETS if args.dataset == "all" else (args.dataset,)
    for dataset in selected:
        run(dataset, args.device, args.output_dir)


if __name__ == "__main__":
    main()
