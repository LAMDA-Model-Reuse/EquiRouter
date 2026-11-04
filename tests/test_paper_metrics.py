import numpy as np

from utils.metrics import evaluate_budget_curve


def test_budget_curve_reports_qnc_saving_and_shared_auc():
    true_performance = np.ones((2, 2), dtype=np.float32)
    true_cost = np.array([[2.0, 1.0], [2.0, 1.0]], dtype=np.float32)
    predicted_performance = np.array([[0.5, 1.0], [0.5, 1.0]], dtype=np.float32)
    predicted_cost = true_cost.copy()

    result = evaluate_budget_curve(
        predicted_performance,
        predicted_cost,
        true_performance,
        true_cost,
        model_names=["premium", "efficient"],
    )

    assert result["peak"] == 1.0
    assert result["qnc"] == 0.5
    assert result["cost_saving_percent_at_target"] == 50.0
    assert result["normalized_auc"] == 1.0
    assert result["auc_cost_bounds"] == [1.0, 2.0]
    assert result["best_model"] == "premium"


def test_unreachable_target_leaves_qnc_undefined():
    true_performance = np.array([[1.0, 0.0], [1.0, 0.0]])
    true_cost = np.array([[2.0, 1.0], [2.0, 1.0]])
    # The expensive model is always scored below the cheap model, including at
    # the largest budget, so the best-single-model target is never reached.
    predicted_performance = np.array([[0.0, 1.0], [0.0, 1.0]])
    predicted_cost = true_cost.copy()

    result = evaluate_budget_curve(
        predicted_performance,
        predicted_cost,
        true_performance,
        true_cost,
    )

    assert result["qnc"] is None
    assert result["qnc_cost"] is None
    assert result["cost_saving_percent_at_target"] is None
