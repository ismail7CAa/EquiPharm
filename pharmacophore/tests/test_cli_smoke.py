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
fake_matching_screening = types.ModuleType("pharmacophore.core.matching_screening")
fake_matching_screening.screen_actives_decoys_matching = Mock()
sys.modules.setdefault("pharmacophore.core.matching_screening", fake_matching_screening)

from pharmacophore.EquiPharm import cli as equipharm_cli
from pharmacophore.EquiPharm import screening as equipharm_screening
from pharmacophore.EquiPharm_Hungarian import cli as hungarian_cli
from pharmacophore.EquiPharm_Hungarian import screening as hungarian_screening
from pharmacophore.EquiPharm_Hungarian_v2 import cli as hungarian_v2_cli
from pharmacophore.EquiPharm_Hungarian_v2 import screening as hungarian_v2_screening
from pharmacophore.Equiformer_with_optimization import cli as equiformer_cli
from pharmacophore.Equiformer_with_optimization import screening as equiformer_screening
from pharmacophore.CDPKit import cli as cdpkit_cli
from pharmacophore.CDPKit import screening as cdpkit_screening
from pharmacophore.Pharmit import cli as pharmit_cli
from pharmacophore.Pharmit import screening as pharmit_screening
from pharmacophore.PharmacoMatch import cli as pharmacomatch_cli
from pharmacophore.PharmacoMatch import screening as pharmacomatch_screening
from pharmacophore import run_all_screening


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

    def test_equipharm_hungarian_wrapper_sets_expected_defaults(self):
        with patch.object(hungarian_screening, "screen_actives_decoys_matching") as run:
            run.return_value = {"roc_auc": 1.0}
            result = hungarian_screening.run_equipharm_hungarian_screening(
                checkpoint_path="checkpoint.pt",
                query_ligand="query.mol2",
                actives_dir="actives_sdf",
                decoys_dir="decoys_sdf",
                output_dir="pharmacophore/results/EquiPharm_Hungarian/aces",
            )

        self.assertEqual(result, {"roc_auc": 1.0})
        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["pipeline_name"], "EquiPharm_Hungarian")
        self.assertEqual(kwargs["matching_method"], "hungarian")
        self.assertEqual(kwargs["model_module"], "benchmarking.Methods.equiformer_encoder_matching")

    def test_equipharm_hungarian_v2_wrapper_sets_expected_defaults(self):
        with patch.object(hungarian_v2_screening, "screen_actives_decoys_matching") as run:
            run.return_value = {"roc_auc": 1.0}
            result = hungarian_v2_screening.run_equipharm_hungarian_v2_screening(
                checkpoint_path="checkpoint.pt",
                query_ligand="query.mol2",
                actives_dir="actives_sdf",
                decoys_dir="decoys_sdf",
                output_dir="pharmacophore/results/EquiPharm_Hungarian_v2/aces",
            )

        self.assertEqual(result, {"roc_auc": 1.0})
        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["pipeline_name"], "EquiPharm_Hungarian_v2")
        self.assertEqual(kwargs["matching_method"], "hungarian")
        self.assertEqual(kwargs["matching_score_mode"], "balanced")
        self.assertEqual(kwargs["model_module"], "benchmarking.Methods.equiformer_encoder_matching")

    def test_all_runner_uses_dataset_specific_output_root(self):
        self.assertEqual(
            run_all_screening.resolve_output_root(Path("pharmacophore/results"), "DUD-E"),
            Path("pharmacophore/results/DUD-E"),
        )
        self.assertEqual(
            run_all_screening.resolve_output_root(Path("pharmacophore/results/DUD-E"), "DUD-E"),
            Path("pharmacophore/results/DUD-E"),
        )

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

    def test_cdpkit_wrapper_builds_expected_commands(self):
        candidates = [(Path("active_0001.sdf"), 1), (Path("decoy_0001.sdf"), 0)]
        with (
            patch.object(cdpkit_screening, "collect_labeled_sdf_files", return_value=candidates),
            patch.object(cdpkit_screening, "write_combined_sdf") as write_sdf,
            patch.object(cdpkit_screening, "run_command") as run_command,
            patch.object(cdpkit_screening, "parse_hit_scores", return_value={"active_0001": 2.0}),
            patch.object(cdpkit_screening, "write_baseline_outputs", return_value={"roc_auc": 1.0}) as write_outputs,
        ):
            result = cdpkit_screening.run_cdpkit_screening(
                query_pharmacophore="query.cdf",
                actives_dir="actives_sdf",
                decoys_dir="decoys_sdf",
                output_dir="pharmacophore/results/CDPKit/aces",
                target_name="aces",
            )

        self.assertEqual(result, {"roc_auc": 1.0})
        self.assertTrue(write_sdf.called)
        self.assertEqual(run_command.call_count, 2)
        rows = write_outputs.call_args.args[1]
        self.assertEqual(rows[0]["pipeline"], "CDPKit")
        self.assertEqual(rows[0]["score"], 2.0)
        self.assertEqual(rows[1]["score"], 0.0)

    def test_cdpkit_dataset_wrapper_writes_summary(self):
        targets = [Path("data/DUD-E/aces"), Path("data/DUD-E/egfr")]
        with (
            patch.object(cdpkit_screening, "discover_dude_targets", return_value=targets),
            patch.object(cdpkit_screening, "find_cdpkit_query", side_effect=[Path("data/DUD-E/aces/query.cdf"), Path("data/DUD-E/egfr/query.cdf")]),
            patch.object(cdpkit_screening, "run_cdpkit_screening", side_effect=[{"pipeline": "CDPKit", "target": "aces"}, {"pipeline": "CDPKit", "target": "egfr"}]) as run,
            patch.object(cdpkit_screening, "write_dataset_summary", return_value=Path("pharmacophore/results/CDPKit/dataset_metrics.csv")) as write_summary,
        ):
            result = cdpkit_screening.run_cdpkit_dataset_screening(
                dataset_dir="data/DUD-E",
                output_dir="pharmacophore/results/CDPKit",
            )

        self.assertEqual(result["n_targets"], 2)
        self.assertEqual(result["n_ok"], 2)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args_list[0].kwargs["output_dir"], Path("pharmacophore/results/CDPKit") / "aces")
        self.assertTrue(write_summary.called)

    def test_pharmacomatch_wrapper_uses_command_template(self):
        candidates = [(Path("active_0001.sdf"), 1)]
        completed = Mock(stdout='{"score": 0.75}', stderr="")
        with (
            patch.object(pharmacomatch_screening, "collect_labeled_sdf_files", return_value=candidates),
            patch.object(pharmacomatch_screening, "run_command", return_value=completed) as run_command,
            patch.object(
                pharmacomatch_screening,
                "write_baseline_outputs",
                return_value={"roc_auc": 1.0},
            ) as write_outputs,
        ):
            result = pharmacomatch_screening.run_pharmacomatch_screening(
                command_template="python screen.py --query {query_ligand} --candidate {candidate}",
                score_json_key="score",
                query_ligand="query.mol2",
                actives_dir="actives_sdf",
                decoys_dir="decoys_sdf",
                output_dir="pharmacophore/results/PharmacoMatch/aces",
                target_name="aces",
            )

        self.assertEqual(result, {"roc_auc": 1.0})
        self.assertEqual(run_command.call_args.args[0], ["python", "screen.py", "--query", "query.mol2", "--candidate", "active_0001.sdf"])
        rows = write_outputs.call_args.args[1]
        self.assertEqual(rows[0]["pipeline"], "PharmacoMatch")
        self.assertEqual(rows[0]["score"], 0.75)

    def test_command_template_baseline_wrapper_sets_pipeline_name(self):
        with patch.object(pharmit_screening, "run_command_baseline_screening") as run:
            run.return_value = {"roc_auc": 1.0}
            result = pharmit_screening.run_pharmit_screening(
                command_template="pharmit --query {query_ligand} --candidate {candidate}",
                score_json_key="score",
                query_ligand="query.mol2",
                actives_dir="actives_sdf",
                decoys_dir="decoys_sdf",
                output_dir="pharmacophore/results/Pharmit/aces",
                target_name="aces",
            )

        self.assertEqual(result, {"roc_auc": 1.0})
        self.assertEqual(run.call_args.kwargs["pipeline_name"], "Pharmit")

    def test_pharmacomatch_dataset_wrapper_writes_summary(self):
        targets = [Path("data/DUD-E/aces"), Path("data/DUD-E/egfr")]
        with (
            patch.object(pharmacomatch_screening, "discover_dude_targets", return_value=targets),
            patch.object(pharmacomatch_screening, "run_pharmacomatch_screening", side_effect=[{"pipeline": "PharmacoMatch", "target": "aces"}, {"pipeline": "PharmacoMatch", "target": "egfr"}]) as run,
            patch.object(pharmacomatch_screening, "write_dataset_summary", return_value=Path("pharmacophore/results/PharmacoMatch/dataset_metrics.csv")) as write_summary,
        ):
            result = pharmacomatch_screening.run_pharmacomatch_dataset_screening(
                dataset_dir="data/DUD-E",
                output_dir="pharmacophore/results/PharmacoMatch",
                command_template="python screen.py --query {query_ligand} --candidate {candidate}",
                score_json_key="score",
            )

        self.assertEqual(result["n_targets"], 2)
        self.assertEqual(result["n_ok"], 2)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args_list[0].kwargs["output_dir"], Path("pharmacophore/results/PharmacoMatch") / "aces")
        self.assertTrue(write_summary.called)


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

    def test_equipharm_hungarian_cli_reads_config_without_data_or_checkpoint(self):
        config = {
            "checkpoint_path": "checkpoints/equipharm/best_model.pt",
            "query_ligand": "data/DUD-E/aces/crystal_ligand.mol2",
            "actives_dir": "data/DUD-E/aces/actives_sdf",
            "decoys_dir": "data/DUD-E/aces/decoys_sdf",
            "output_dir": "pharmacophore/results/EquiPharm_Hungarian/aces",
            "target_name": "aces",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "target.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            argv = ["cli", "--config", str(config_path), "--limit", "5"]
            with (
                patch.object(sys, "argv", argv),
                patch.object(sys, "stdout", StringIO()),
                patch.object(hungarian_cli, "run_equipharm_hungarian_screening") as run,
            ):
                run.return_value = {"pipeline": "EquiPharm_Hungarian", "target": "aces"}
                hungarian_cli.main()

        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["target_name"], "aces")
        self.assertEqual(kwargs["limit"], 5)

    def test_equipharm_hungarian_v2_cli_reads_config_without_data_or_checkpoint(self):
        config = {
            "checkpoint_path": "checkpoints/equipharm/best_model.pt",
            "query_ligand": "data/DUD-E/aces/crystal_ligand.mol2",
            "actives_dir": "data/DUD-E/aces/actives_sdf",
            "decoys_dir": "data/DUD-E/aces/decoys_sdf",
            "output_dir": "pharmacophore/results/EquiPharm_Hungarian_v2/aces",
            "target_name": "aces",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "target.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            argv = ["cli", "--config", str(config_path), "--limit", "5"]
            with (
                patch.object(sys, "argv", argv),
                patch.object(sys, "stdout", StringIO()),
                patch.object(hungarian_v2_cli, "run_equipharm_hungarian_v2_screening") as run,
            ):
                run.return_value = {"pipeline": "EquiPharm_Hungarian_v2", "target": "aces"}
                hungarian_v2_cli.main()

        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["target_name"], "aces")
        self.assertEqual(kwargs["limit"], 5)
        self.assertEqual(kwargs["output_dir"], "pharmacophore/results/EquiPharm_Hungarian_v2/aces")

    def test_cdpkit_cli_reads_config_without_running_external_tools(self):
        config = {
            "query_pharmacophore": "data/CDPKit/aces/query.cdf",
            "actives_dir": "data/DUD-E/aces/actives_sdf",
            "decoys_dir": "data/DUD-E/aces/decoys_sdf",
            "output_dir": "pharmacophore/results/CDPKit/aces",
            "target_name": "aces",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "target.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            argv = ["cli", "--config", str(config_path), "--limit", "5"]
            with (
                patch.object(sys, "argv", argv),
                patch.object(sys, "stdout", StringIO()),
                patch.object(cdpkit_cli, "run_cdpkit_screening") as run,
            ):
                run.return_value = {"pipeline": "CDPKit", "target": "aces"}
                cdpkit_cli.main()

        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["target_name"], "aces")
        self.assertEqual(kwargs["limit"], 5)

    def test_command_template_cli_reads_config_without_running_external_tool(self):
        config = {
            "command_template": "pharmit --query {query_ligand} --candidate {candidate}",
            "score_json_key": "score",
            "query_ligand": "data/DUD-E/aces/crystal_ligand.mol2",
            "actives_dir": "data/DUD-E/aces/actives_sdf",
            "decoys_dir": "data/DUD-E/aces/decoys_sdf",
            "output_dir": "pharmacophore/results/Pharmit/aces",
            "target_name": "aces",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "target.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            argv = ["cli", "--config", str(config_path), "--limit", "5"]
            with (
                patch.object(sys, "argv", argv),
                patch.object(sys, "stdout", StringIO()),
                patch.object(pharmit_cli, "run_pharmit_screening") as run,
            ):
                run.return_value = {"pipeline": "Pharmit", "target": "aces"}
                pharmit_cli.main()

        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["target_name"], "aces")
        self.assertEqual(kwargs["limit"], 5)
        self.assertEqual(kwargs["output_dir"], "pharmacophore/results/Pharmit/aces")

    def test_cdpkit_cli_reads_dataset_config_without_running_external_tools(self):
        config = {
            "dataset_dir": "data/DUD-E",
            "output_dir": "pharmacophore/results/CDPKit",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "dataset.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            argv = ["cli", "--config", str(config_path), "--limit", "5"]
            with (
                patch.object(sys, "argv", argv),
                patch.object(sys, "stdout", StringIO()),
                patch.object(cdpkit_cli, "run_cdpkit_dataset_screening") as run,
            ):
                run.return_value = {"pipeline": "CDPKit", "n_targets": 2}
                cdpkit_cli.main()

        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["dataset_dir"], "data/DUD-E")
        self.assertEqual(kwargs["limit"], 5)
        self.assertEqual(kwargs["output_dir"], "pharmacophore/results/CDPKit")

    def test_pharmacomatch_cli_reads_config_without_running_external_tools(self):
        config = {
            "command_template": "python screen.py --query {query_ligand} --candidate {candidate}",
            "score_json_key": "score",
            "query_ligand": "data/DUD-E/aces/crystal_ligand.mol2",
            "actives_dir": "data/DUD-E/aces/actives_sdf",
            "decoys_dir": "data/DUD-E/aces/decoys_sdf",
            "output_dir": "pharmacophore/results/PharmacoMatch/aces",
            "target_name": "aces",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "target.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            argv = ["cli", "--config", str(config_path), "--limit", "5"]
            with (
                patch.object(sys, "argv", argv),
                patch.object(sys, "stdout", StringIO()),
                patch.object(pharmacomatch_cli, "run_pharmacomatch_screening") as run,
            ):
                run.return_value = {"pipeline": "PharmacoMatch", "target": "aces"}
                pharmacomatch_cli.main()

        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["target_name"], "aces")
        self.assertEqual(kwargs["limit"], 5)
        self.assertEqual(kwargs["output_dir"], "pharmacophore/results/PharmacoMatch/aces")

    def test_pharmacomatch_cli_reads_dataset_config_without_running_external_tools(self):
        config = {
            "dataset_dir": "data/DUD-E",
            "output_dir": "pharmacophore/results/PharmacoMatch",
            "command_template": "python screen.py --query {query_ligand} --candidate {candidate}",
            "score_json_key": "score",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "dataset.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            argv = ["cli", "--config", str(config_path), "--limit", "5"]
            with (
                patch.object(sys, "argv", argv),
                patch.object(sys, "stdout", StringIO()),
                patch.object(pharmacomatch_cli, "run_pharmacomatch_dataset_screening") as run,
            ):
                run.return_value = {"pipeline": "PharmacoMatch", "n_targets": 2}
                pharmacomatch_cli.main()

        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["dataset_dir"], "data/DUD-E")
        self.assertEqual(kwargs["limit"], 5)
        self.assertEqual(kwargs["output_dir"], "pharmacophore/results/PharmacoMatch")


if __name__ == "__main__":
    unittest.main()
