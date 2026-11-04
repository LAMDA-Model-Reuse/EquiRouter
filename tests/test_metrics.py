import unittest

import numpy as np

from utils.metrics import (
    extract_pareto_front,
    minimum_cost_policy_point,
    normalized_auc,
)


class ParetoAndAUCTest(unittest.TestCase):
    def test_pareto_deduplicates_cost_and_removes_dominated_points(self):
        points = [
            {"cost": 0.2, "performance": 0.4},
            {"cost": 0.2, "performance": 0.5},
            {"cost": 0.5, "performance": 0.45},
            {"cost": 0.8, "performance": 0.9},
        ]
        self.assertEqual(
            extract_pareto_front(points),
            [
                {"cost": 0.2, "performance": 0.5},
                {"cost": 0.8, "performance": 0.9},
            ],
        )

    def test_observed_range_auc_uses_both_endpoints(self):
        points = [
            {"cost": 0.2, "performance": 0.5},
            {"cost": 0.8, "performance": 0.9},
        ]
        self.assertAlmostEqual(normalized_auc(points), 0.7)

    def test_shared_range_uses_explicit_lower_boundary_and_right_extension(self):
        points = [
            {"cost": 0.2, "performance": 0.5},
            {"cost": 0.8, "performance": 0.9},
        ]
        self.assertAlmostEqual(normalized_auc(points, cost_bounds=(0.0, 1.0)), 0.65)

    def test_degenerate_cost_range_returns_best_feasible_performance(self):
        points = [
            {"cost": 0.5, "performance": 0.4},
            {"cost": 0.5, "performance": 0.8},
        ]
        self.assertAlmostEqual(normalized_auc(points, cost_bounds=(0.5, 0.5)), 0.8)


class TradeoffConstructionTest(unittest.TestCase):
    def test_minimum_cost_anchor_uses_matching_realized_performance(self):
        point = minimum_cost_policy_point(
            perf_mat=np.array([[0.2, 0.9], [0.8, 0.1]]),
            cost_mat=np.array([[0.4, 0.1], [0.2, 0.7]]),
        )
        self.assertAlmostEqual(point["cost"], 0.15)
        self.assertAlmostEqual(point["performance"], 0.85)


if __name__ == "__main__":
    unittest.main()
