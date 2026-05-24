"""Tests for screening metric exports."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from unittest.mock import patch

from pharmacophore.core.metrics import bedroc, enrichment_factor, write_outputs


class ScreeningMetricsTests(unittest.TestCase):
    def test_write_outputs_exports_early_recognition_metrics_and_roc_coordinates(self):
        rows = [
            {"pipeline": "EquiPharm", "target": "aces", "name": "active_a", "label": 1, "score": 0.95},
            {"pipeline": "EquiPharm", "target": "aces", "name": "decoy_a", "label": 0, "score": 0.70},
            {"pipeline": "EquiPharm", "target": "aces", "name": "active_b", "label": 1, "score": 0.60},
            {"pipeline": "EquiPharm", "target": "aces", "name": "decoy_b", "label": 0, "score": 0.20},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            with (
                patch("pharmacophore.core.metrics.plot_score_boxplot"),
                patch("pharmacophore.core.metrics.plot_roc_curve"),
            ):
                metrics = write_outputs(
                    output_dir,
                    rows,
                    pipeline_name="EquiPharm",
                    target_name="aces",
                )

            summary = pd.read_csv(output_dir / "screening_performance_summary.csv")
            roc_coordinates = pd.read_csv(output_dir / "auroc_curve_coordinates.csv")

        self.assertIn("ef1_percent", metrics)
        self.assertIn("bedroc_alpha20", metrics)
        self.assertEqual(float(summary.loc[0, "roc_auc"]), metrics["roc_auc"])
        self.assertEqual(float(summary.loc[0, "ef1_percent"]), metrics["ef1_percent"])
        self.assertEqual(float(summary.loc[0, "bedroc_alpha20"]), metrics["bedroc_alpha20"])
        self.assertEqual(
            list(roc_coordinates.columns),
            ["false_positive_rate", "true_positive_rate", "threshold"],
        )
        self.assertGreaterEqual(len(roc_coordinates), 2)

    def test_early_recognition_metrics_reward_top_ranked_actives(self):
        labels = [1, 0, 1, 0]
        strong_scores = [0.95, 0.70, 0.60, 0.20]
        weak_scores = [0.20, 0.95, 0.60, 0.70]

        self.assertGreater(enrichment_factor(strong_scores, labels, fraction=0.25), 1.0)
        self.assertGreater(bedroc(strong_scores, labels), bedroc(weak_scores, labels))


if __name__ == "__main__":
    unittest.main()
