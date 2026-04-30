"""Smoke tests for pharmacophore screening entry points."""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

fake_screening = types.ModuleType("pharmacophore.core.screening")
fake_screening.screen_actives_decoys = Mock()
sys.modules.setdefault("pharmacophore.core.screening", fake_screening)

from pharmacophore.EquiPharm import cli as equipharm_cli
from pharmacophore.EquiPharm import screening as equipharm_screening
from pharmacophore.Equiformer_with_optimization import cli as equiformer_cli
from pharmacophore.Equiformer_with_optimization import screening as equiformer_screening


class PipelineWrapperTests(unittest.TestCase):
    def test_equipharm_wrapper_sets_expected_defaults(self):
        with patch.object(equipharm_screening, "screen_actives_decoys") as run:
            run.return_value = {"roc_auc": 1.0}
            result = equipharm_screening.run_equipharm_screening(
                checkpoint_path="checkpoint.pt",
                query_ligand="query.mol2",
                actives_dir="actives_sdf",
                decoys_dir="decoys_sdf",
                output_dir="pharmacophore/results/EquiPharm/aces",
            )

        self.assertEqual(result, {"roc_auc": 1.0})
        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["pipeline_name"], "EquiPharm")
        self.assertEqual(kwargs["model_module"], "benchmarking.Methods.equiformer_encoder_pharmaco_feat")
        self.assertTrue(kwargs["use_pharmacophore_features"])

    def test_equiformer_wrapper_sets_expected_defaults(self):
        with patch.object(equiformer_screening, "screen_actives_decoys") as run:
            run.return_value = {"roc_auc": 0.5}
            result = equiformer_screening.run_equiformer_optimization_screening(
                checkpoint_path="checkpoint.pt",
                query_ligand="query.mol2",
                actives_dir="actives_sdf",
                decoys_dir="decoys_sdf",
                output_dir="pharmacophore/results/Equiformer_with_optimization/aces",
            )

        self.assertEqual(result, {"roc_auc": 0.5})
        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["pipeline_name"], "Equiformer_with_optimization")
        self.assertEqual(kwargs["model_module"], "benchmarking.Methods.equiformer_architecture")
        self.assertFalse(kwargs["use_pharmacophore_features"])


class CliConfigTests(unittest.TestCase):
    def test_equipharm_cli_reads_config_without_data_or_checkpoint(self):
        config = {
            "checkpoint_path": "checkpoints/equipharm/best_model.pt",
            "query_ligand": "data/DUD-E/aces/crystal_ligand.mol2",
            "actives_dir": "data/DUD-E/aces/actives_sdf",
            "decoys_dir": "data/DUD-E/aces/decoys_sdf",
            "output_dir": "pharmacophore/results/EquiPharm/aces",
            "target_name": "aces",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "target.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            argv = ["cli", "--config", str(config_path), "--limit", "5"]
            with (
                patch.object(sys, "argv", argv),
                patch.object(sys, "stdout", StringIO()),
                patch.object(equipharm_cli, "run_equipharm_screening") as run,
            ):
                run.return_value = {"pipeline": "EquiPharm", "target": "aces"}
                equipharm_cli.main()

        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["target_name"], "aces")
        self.assertEqual(kwargs["limit"], 5)
        self.assertEqual(kwargs["output_dir"], "pharmacophore/results/EquiPharm/aces")

    def test_equiformer_cli_reads_config_without_data_or_checkpoint(self):
        config = {
            "checkpoint_path": "checkpoints/equiformer/best_model.pt",
            "query_ligand": "data/DUD-E/aces/crystal_ligand.mol2",
            "actives_dir": "data/DUD-E/aces/actives_sdf",
            "decoys_dir": "data/DUD-E/aces/decoys_sdf",
            "output_dir": "pharmacophore/results/Equiformer_with_optimization/aces",
            "target_name": "aces",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "target.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            argv = ["cli", "--config", str(config_path), "--limit", "5"]
            with (
                patch.object(sys, "argv", argv),
                patch.object(sys, "stdout", StringIO()),
                patch.object(equiformer_cli, "run_equiformer_optimization_screening") as run,
            ):
                run.return_value = {"pipeline": "Equiformer_with_optimization", "target": "aces"}
                equiformer_cli.main()

        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["target_name"], "aces")
        self.assertEqual(kwargs["limit"], 5)
        self.assertEqual(kwargs["output_dir"], "pharmacophore/results/Equiformer_with_optimization/aces")


if __name__ == "__main__":
    unittest.main()
