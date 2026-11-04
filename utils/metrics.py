"""Shared performance-cost curve utilities.

The normalized AUC in this module uses a caller-supplied, benchmark-wide cost
interval.  This is important: normalizing every router by its own maximum cost
makes scores from different routers incomparable.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

Point = Mapping[str, float]


def minimum_cost_policy_point(perf_mat, cost_mat) -> dict[str, float]:
    """Evaluate the common policy that picks each row's cheapest actual model."""
    perf_mat = np.asarray(perf_mat, dtype=np.float64)
    cost_mat = np.asarray(cost_mat, dtype=np.float64)
    if perf_mat.shape != cost_mat.shape or perf_mat.ndim != 2:
        raise ValueError(
            "Performance and cost matrices must be matching 2D arrays; "
            f"got {perf_mat.shape} and {cost_mat.shape}"
        )
    if not np.isfinite(cost_mat).all():
        raise ValueError("Ground-truth evaluation costs must all be finite.")

    rows = np.arange(cost_mat.shape[0])
    choices = np.argmin(cost_mat, axis=1)
    selected_performance = perf_mat[rows, choices]
    selected_cost = cost_mat[rows, choices]
    if not np.isfinite(selected_performance).all():
        raise ValueError("Minimum-cost policy selected non-finite performance values.")
    return {
        "cost": float(np.mean(selected_cost)),
        "performance": float(np.mean(selected_performance)),
    }


def _finite_points(points: Iterable[Point]) -> list[dict[str, float]]:
    cleaned: list[dict[str, float]] = []
    for point in points:
        cost = float(point["cost"])
        performance = float(point["performance"])
        if np.isfinite(cost) and np.isfinite(performance):
            cleaned.append({"cost": cost, "performance": performance})
    return cleaned


def extract_pareto_front(points: Iterable[Point]) -> list[dict[str, float]]:
    """Return the non-dominated performance envelope in ascending cost order.

    At an identical cost only the highest-performance point is retained.
    """
    cleaned = _finite_points(points)
    if not cleaned:
        return []

    best_at_cost: dict[float, float] = {}
    for point in cleaned:
        cost = point["cost"]
        best_at_cost[cost] = max(best_at_cost.get(cost, -np.inf), point["performance"])

    pareto_front: list[dict[str, float]] = []
    best_performance = -np.inf
    for cost in sorted(best_at_cost):
        performance = best_at_cost[cost]
        if performance > best_performance:
            pareto_front.append({"cost": cost, "performance": performance})
            best_performance = performance
    return pareto_front


def normalized_auc(
    points: Sequence[Point],
    cost_bounds: tuple[float, float] | None = None,
) -> float:
    """Calculate normalized AUC on a fixed cost interval.

    The curve is the Pareto performance envelope.  If no evaluated policy is
    affordable at the lower bound, quality starts at zero.  Benchmark callers
    should normally include a real minimum-cost policy point at that boundary.
    Beyond the last evaluated point, the last policy remains usable, so its
    performance is carried to the upper bound.

    When ``cost_bounds`` is omitted, the observed cost range is used.  Benchmark
    evaluation should always pass shared bounds so routers remain comparable.
    """
    cleaned = _finite_points(points)
    if not cleaned:
        return 0.0

    if cost_bounds is None:
        lower = min(point["cost"] for point in cleaned)
        upper = max(point["cost"] for point in cleaned)
    else:
        lower, upper = (float(cost_bounds[0]), float(cost_bounds[1]))

    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError(f"Cost bounds must be finite, got {(lower, upper)}")
    if upper < lower:
        raise ValueError(f"Cost bounds must be ordered, got {(lower, upper)}")

    pareto = extract_pareto_front(cleaned)
    if upper == lower:
        feasible = [p["performance"] for p in pareto if p["cost"] <= upper]
        return float(max(feasible, default=0.0))

    affordable_at_lower = [point["performance"] for point in pareto if point["cost"] <= lower]
    current_performance = float(max(affordable_at_lower, default=0.0))
    curve_costs = [lower]
    curve_performance = [current_performance]

    for point in pareto:
        cost = point["cost"]
        performance = point["performance"]
        if cost <= lower or cost >= upper:
            continue
        if performance > current_performance:
            curve_costs.append(cost)
            curve_performance.append(performance)
            current_performance = performance

    affordable_at_upper = [point["performance"] for point in pareto if point["cost"] <= upper]
    if affordable_at_upper:
        current_performance = max(current_performance, max(affordable_at_upper))

    curve_costs.append(upper)
    curve_performance.append(current_performance)

    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    area = trapezoid(
        np.asarray(curve_performance, dtype=np.float64),
        np.asarray(curve_costs, dtype=np.float64),
    )
    return float(area / (upper - lower))


def evaluate_budget_curve(
    predicted_performance,
    predicted_cost,
    true_performance,
    true_cost,
    *,
    model_names: Sequence[str] | None = None,
    budget_points: int = 100,
) -> dict:
    """Evaluate Peak, QNC, cost saving, and nAUC on one shared protocol.

    The grid contains ``budget_points`` equally spaced thresholds from
    ``max_q min_j predicted_cost[q, j]`` to ``max_q,j predicted_cost[q, j]``.
    The lower bound ensures that every query has a feasible candidate. QNC is
    undefined when no evaluated policy reaches the best-single-model target.
    """
    predicted_performance = np.asarray(predicted_performance, dtype=np.float64)
    predicted_cost = np.asarray(predicted_cost, dtype=np.float64)
    true_performance = np.asarray(true_performance, dtype=np.float64)
    true_cost = np.asarray(true_cost, dtype=np.float64)
    shapes = {
        predicted_performance.shape,
        predicted_cost.shape,
        true_performance.shape,
        true_cost.shape,
    }
    if len(shapes) != 1 or predicted_performance.ndim != 2:
        raise ValueError("All prediction and target matrices must share shape (N, M)")
    if 0 in predicted_performance.shape:
        raise ValueError("Budget evaluation requires queries and candidate models")
    if model_names is not None and len(model_names) != predicted_performance.shape[1]:
        raise ValueError("model_names must match the number of candidate columns")
    if budget_points <= 0:
        raise ValueError("budget_points must be positive")
    if not all(
        np.isfinite(values).all()
        for values in (
            predicted_performance,
            predicted_cost,
            true_performance,
            true_cost,
        )
    ):
        raise ValueError("Budget-curve inputs must be finite")
    if (predicted_cost < 0).any() or (true_cost < 0).any():
        raise ValueError("Evaluation costs must be non-negative")

    minimum_budget = float(np.max(np.min(predicted_cost, axis=1)))
    maximum_budget = float(np.max(predicted_cost))
    budgets = np.linspace(minimum_budget, maximum_budget, budget_points)
    rows = np.arange(predicted_performance.shape[0])
    curve: list[dict[str, float]] = []
    for budget in budgets:
        feasible = predicted_cost <= budget
        selected = np.argmax(np.where(feasible, predicted_performance, -np.inf), axis=1)
        curve.append(
            {
                "budget": float(budget),
                "cost": float(true_cost[rows, selected].mean()),
                "performance": float(true_performance[rows, selected].mean()),
            }
        )

    single_model_performance = true_performance.mean(axis=0)
    single_model_cost = true_cost.mean(axis=0)
    best_model_id = int(np.argmax(single_model_performance))
    target_performance = float(single_model_performance[best_model_id])
    target_cost = float(single_model_cost[best_model_id])
    qualifying = [point for point in curve if point["performance"] >= target_performance]
    qnc_cost = min((point["cost"] for point in qualifying), default=None)
    qnc = None if qnc_cost is None or target_cost == 0 else float(qnc_cost / target_cost)
    peak_point = max(curve, key=lambda point: point["performance"])
    minimum_cost_point = minimum_cost_policy_point(true_performance, true_cost)
    cost_bounds = (
        float(np.mean(np.min(true_cost, axis=1))),
        float(np.mean(np.max(true_cost, axis=1))),
    )
    pareto_curve = extract_pareto_front([minimum_cost_point, *curve])
    auc = normalized_auc(pareto_curve, cost_bounds=cost_bounds)
    saving = None if qnc is None else float(1.0 - qnc)
    names = list(model_names) if model_names is not None else None

    return {
        "peak": peak_point["performance"],
        "peak_cost": peak_point["cost"],
        "peak_budget": peak_point["budget"],
        "qnc": qnc,
        "qnc_cost": qnc_cost,
        "cost_saving_rate_at_target": saving,
        "cost_saving_percent_at_target": None if saving is None else 100.0 * saving,
        "normalized_auc": auc,
        "auc_cost_bounds": list(cost_bounds),
        "minimum_cost_policy": minimum_cost_point,
        "pareto_curve": pareto_curve,
        "target_performance": target_performance,
        "target_cost": target_cost,
        "best_model_id": best_model_id,
        "best_model": None if names is None else names[best_model_id],
        "curve": curve,
    }
