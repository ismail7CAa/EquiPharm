"""Tests for screening metric exports."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from unittest.mock import patch

from pharmacophore.core.metrics import bedroc, enrichment_factor, write_outputs
from pharmacophore.core.resume import append_score_row, initialize_score_file, load_resume_rows

try:
    import torch
    from pharmacophore.core.matching import hungarian_matching_score, matching_score
except ModuleNotFoundError:
    torch = None
    hungarian_matching_score = None
    matching_score = None


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

    def test_resume_rows_track_only_completed_finite_scores(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            append_score_row(
                output_dir,
                {
                    "pipeline": "EquiPharm",
                    "target": "aces",
                    "name": "active_00000",
                    "path": "data/DUD-E/aces/actives_sdf/a.sdf",
                    "label": 1,
                    "score": 0.8,
                },
            )
            append_score_row(
                output_dir,
                {
                    "pipeline": "EquiPharm",
                    "target": "aces",
                    "name": "bad",
                    "path": "data/DUD-E/aces/decoys_sdf/bad.sdf",
                    "label": 0,
                    "score": float("nan"),
                    "error": "failed",
                },
            )

            rows, completed_paths = load_resume_rows(output_dir)

        self.assertEqual(len(rows), 2)
        self.assertEqual(completed_paths, {"data/DUD-E/aces/actives_sdf/a.sdf"})

    def test_initialize_score_file_writes_header_before_first_candidate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            score_path = initialize_score_file(tmpdir, ["path", "score"])
            rows, completed_paths = load_resume_rows(tmpdir)

        self.assertEqual(score_path.name, "scores.csv")
        self.assertEqual(rows, [])
        self.assertEqual(completed_paths, set())

    def test_hungarian_matching_uses_one_to_one_assignments(self):
        if torch is None:
            self.skipTest("torch is not installed")
        similarity = torch.tensor(
            [
                [0.9, 0.2],
                [0.8, 0.7],
            ],
            dtype=torch.float32,
        )

        score, assignments, components = hungarian_matching_score(similarity)
        self.assertAlmostEqual(score, 0.8, places=6)
        self.assertEqual(assignments, [(0, 0), (1, 1)])
        self.assertAlmostEqual(components["strict_score"], 0.8, places=6)

    def test_matching_score_forbids_feature_family_mismatch(self):
        if torch is None:
            self.skipTest("torch is not installed")
        query = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        candidate = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        query_metadata = [{"family": "Donor"}, {"family": "Aromatic"}]
        candidate_metadata = [{"family": "Aromatic"}, {"family": "Donor"}]

        score, similarity, match_details, components = matching_score(
            query,
            candidate,
            query_metadata=query_metadata,
            candidate_metadata=candidate_metadata,
            method="hungarian",
        )

        self.assertEqual(tuple(similarity.shape), (2, 2))
        self.assertAlmostEqual(score, 0.0, places=6)
        self.assertAlmostEqual(components["balanced_score"], 0.0, places=6)
        self.assertEqual([match["status"] for match in match_details], ["unmatched", "unmatched"])
        self.assertEqual(match_details[0]["query_family"], "Donor")
        self.assertIsNone(match_details[0]["candidate_index"])

    def test_feature_distance_score_uses_negative_average_matched_distance(self):
        if torch is None:
            self.skipTest("torch is not installed")
        query = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        candidate = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        query_metadata = [
            {"family": "Donor", "center": (0.0, 0.0, 0.0)},
            {"family": "Acceptor", "center": (0.0, 2.0, 0.0)},
        ]
        candidate_metadata = [
            {"family": "Donor", "center": (0.0, 1.0, 0.0)},
            {"family": "Acceptor", "center": (0.0, 4.0, 0.0)},
            {"family": "Aromatic", "center": (9.0, 9.0, 9.0)},
            {"family": "Hydrophobe", "center": (8.0, 8.0, 8.0)},
        ]

        score, _, match_details, components = matching_score(
            query,
            candidate,
            query_metadata=query_metadata,
            candidate_metadata=candidate_metadata,
            method="hungarian",
            score_mode="feature_distance",
        )

        self.assertAlmostEqual(score, -1.5, places=6)
        self.assertAlmostEqual(components["average_feature_distance"], 1.5, places=6)
        self.assertAlmostEqual(components["feature_distance_score"], -1.5, places=6)
        self.assertEqual(components["matched_feature_distance_count"], 2)
        self.assertEqual([match["feature_distance"] for match in match_details], [1.0, 2.0])
        self.assertEqual([match["status"] for match in match_details], ["matched", "matched"])

    def test_embedding_distance_score_uses_negative_average_matched_embedding_distance(self):
        if torch is None:
            self.skipTest("torch is not installed")
        query = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 2.0],
            ],
            dtype=torch.float32,
        )
        candidate = torch.tensor(
            [
                [0.0, 1.0],
                [0.0, 4.0],
            ],
            dtype=torch.float32,
        )
        metadata = [{"family": "Donor"}, {"family": "Acceptor"}]

        score, _, match_details, components = matching_score(
            query,
            candidate,
            query_metadata=metadata,
            candidate_metadata=metadata,
            method="hungarian_euclidean",
            score_mode="embedding_distance",
        )

        self.assertAlmostEqual(score, -1.5, places=6)
        self.assertAlmostEqual(components["average_embedding_distance"], 1.5, places=6)
        self.assertAlmostEqual(components["embedding_distance_score"], -1.5, places=6)
        self.assertEqual(components["matched_embedding_distance_count"], 2)
        self.assertEqual([match["status"] for match in match_details], ["matched", "matched"])

    def test_embedding_geometry_distance_score_compares_internal_embedding_distances(self):
        if torch is None:
            self.skipTest("torch is not installed")
        query = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 2.0],
            ],
            dtype=torch.float32,
        )
        candidate = torch.tensor(
            [
                [0.0, 1.0],
                [0.0, 4.0],
            ],
            dtype=torch.float32,
        )
        metadata = [{"family": "Donor"}, {"family": "Acceptor"}]

        score, _, match_details, components = matching_score(
            query,
            candidate,
            query_metadata=metadata,
            candidate_metadata=metadata,
            method="hungarian_euclidean",
            score_mode="embedding_geometry_distance",
        )

        self.assertAlmostEqual(score, -1.0, places=6)
        self.assertAlmostEqual(components["average_embedding_geometry_delta"], 1.0, places=6)
        self.assertAlmostEqual(components["embedding_geometry_distance_score"], -1.0, places=6)
        self.assertEqual(components["embedding_geometry_pair_count"], 1)
        self.assertEqual([match["status"] for match in match_details], ["matched", "matched"])

    def test_hungarian_euclidean_assigns_by_embedding_distance(self):
        if torch is None:
            self.skipTest("torch is not installed")
        query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        candidate = torch.tensor(
            [
                [2.0, 0.0],
                [0.9, 0.1],
            ],
            dtype=torch.float32,
        )
        metadata = [{"family": "Donor"}]
        candidate_metadata = [{"family": "Donor"}, {"family": "Donor"}]

        score, similarity, match_details, components = matching_score(
            query,
            candidate,
            query_metadata=metadata,
            candidate_metadata=candidate_metadata,
            method="hungarian_euclidean",
            score_mode="embedding_distance",
        )

        self.assertEqual(match_details[0]["candidate_index"], 1)
        self.assertAlmostEqual(float(similarity[0, 0].item()), -1.0, places=6)
        self.assertAlmostEqual(float(similarity[0, 1].item()), -(2 ** 0.5 / 10), places=6)
        self.assertAlmostEqual(score, -(2 ** 0.5 / 10), places=6)

    def test_hungarian_3d_assigns_by_feature_center_distance(self):
        if torch is None:
            self.skipTest("torch is not installed")
        query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        candidate = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        query_metadata = [{"family": "Donor", "center": (0.0, 0.0, 0.0)}]
        candidate_metadata = [
            {"family": "Donor", "center": (10.0, 0.0, 0.0)},
            {"family": "Donor", "center": (1.0, 0.0, 0.0)},
        ]

        score, similarity, match_details, components = matching_score(
            query,
            candidate,
            query_metadata=query_metadata,
            candidate_metadata=candidate_metadata,
            method="hungarian_3d",
            score_mode="feature_distance",
        )

        self.assertEqual(tuple(similarity.shape), (1, 2))
        self.assertAlmostEqual(float(similarity[0, 0].item()), -10.0, places=6)
        self.assertAlmostEqual(float(similarity[0, 1].item()), -1.0, places=6)
        self.assertEqual(match_details[0]["candidate_index"], 1)
        self.assertAlmostEqual(score, -1.0, places=6)
        self.assertAlmostEqual(components["average_feature_distance"], 1.0, places=6)

    def test_geometry_distance_score_compares_internal_matched_feature_distances(self):
        if torch is None:
            self.skipTest("torch is not installed")
        query = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        candidate = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        query_metadata = [
            {"family": "Donor", "center": (0.0, 0.0, 0.0)},
            {"family": "Acceptor", "center": (0.0, 2.0, 0.0)},
        ]
        candidate_metadata = [
            {"family": "Donor", "center": (0.0, 1.0, 0.0)},
            {"family": "Acceptor", "center": (0.0, 4.0, 0.0)},
        ]

        score, _, match_details, components = matching_score(
            query,
            candidate,
            query_metadata=query_metadata,
            candidate_metadata=candidate_metadata,
            method="hungarian",
            score_mode="geometry_distance",
        )

        self.assertAlmostEqual(score, -1.0, places=6)
        self.assertAlmostEqual(components["average_geometry_distance_delta"], 1.0, places=6)
        self.assertAlmostEqual(components["geometry_distance_score"], -1.0, places=6)
        self.assertEqual(components["geometry_distance_pair_count"], 1)
        self.assertEqual([match["status"] for match in match_details], ["matched", "matched"])

    def test_euclidean_embedding_assignment_can_score_3d_geometry(self):
        if torch is None:
            self.skipTest("torch is not installed")
        query = torch.tensor(
            [
                [0.0, 0.0],
                [10.0, 0.0],
            ],
            dtype=torch.float32,
        )
        candidate = torch.tensor(
            [
                [10.1, 0.0],
                [0.1, 0.0],
            ],
            dtype=torch.float32,
        )
        query_metadata = [
            {"family": "Donor", "center": (0.0, 0.0, 0.0)},
            {"family": "Acceptor", "center": (0.0, 2.0, 0.0)},
        ]
        candidate_metadata = [
            {"family": "Acceptor", "center": (0.0, 4.0, 0.0)},
            {"family": "Donor", "center": (0.0, 1.0, 0.0)},
        ]

        score, _, match_details, components = matching_score(
            query,
            candidate,
            query_metadata=query_metadata,
            candidate_metadata=candidate_metadata,
            method="hungarian_euclidean",
            score_mode="geometry_distance",
        )

        self.assertEqual([match["candidate_index"] for match in match_details], [1, 0])
        self.assertAlmostEqual(score, -1.0, places=6)
        self.assertAlmostEqual(components["geometry_distance_score"], -1.0, places=6)
        self.assertAlmostEqual(components["average_geometry_distance_delta"], 1.0, places=6)
        self.assertEqual(components["geometry_distance_pair_count"], 1)

    def test_cosine_score_uses_average_matched_embedding_similarity(self):
        if torch is None:
            self.skipTest("torch is not installed")
        query = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        candidate = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        query_metadata = [{"family": "Donor"}, {"family": "Acceptor"}]
        candidate_metadata = [
            {"family": "Donor"},
            {"family": "Acceptor"},
            {"family": "Aromatic"},
        ]

        score, _, match_details, components = matching_score(
            query,
            candidate,
            query_metadata=query_metadata,
            candidate_metadata=candidate_metadata,
            method="hungarian",
            score_mode="cosine",
        )

        self.assertAlmostEqual(score, 1.0, places=6)
        self.assertAlmostEqual(components["matched_cosine_similarity_score"], 1.0, places=6)
        self.assertEqual(components["matched_cosine_similarity_count"], 2)
        self.assertEqual([match["status"] for match in match_details], ["matched", "matched"])

    def test_cosine_geometry_score_compares_internal_embedding_distances(self):
        if torch is None:
            self.skipTest("torch is not installed")
        query = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        candidate = torch.tensor(
            [
                [1.0, 0.0],
                [0.5, 0.8660254],
            ],
            dtype=torch.float32,
        )
        query_metadata = [{"family": "Donor"}, {"family": "Acceptor"}]
        candidate_metadata = [{"family": "Donor"}, {"family": "Acceptor"}]

        score, _, match_details, components = matching_score(
            query,
            candidate,
            query_metadata=query_metadata,
            candidate_metadata=candidate_metadata,
            method="hungarian",
            score_mode="cosine_geometry",
        )

        self.assertAlmostEqual(score, -0.5, places=6)
        self.assertAlmostEqual(components["average_cosine_geometry_delta"], 0.5, places=6)
        self.assertAlmostEqual(components["cosine_geometry_score"], -0.5, places=6)
        self.assertEqual(components["cosine_geometry_pair_count"], 1)
        self.assertEqual([match["status"] for match in match_details], ["matched", "matched"])


if __name__ == "__main__":
    unittest.main()
