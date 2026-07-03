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
from pharmacophore.EquiPharm_Hungarian_3D import cli as hungarian_3d_cli
from pharmacophore.EquiPharm_Hungarian_3D import screening as hungarian_3d_screening
from pharmacophore.EquiPharm_Hungarian_Cosine import cli as hungarian_cosine_cli
from pharmacophore.EquiPharm_Hungarian_Cosine import screening as hungarian_cosine_screening
from pharmacophore.EquiPharm_Hungarian_Cosine_v2 import cli as hungarian_cosine_v2_cli
from pharmacophore.EquiPharm_Hungarian_Cosine_v2 import screening as hungarian_cosine_v2_screening
from pharmacophore.EquiPharm_Hungarian_v2 import cli as hungarian_v2_cli
from pharmacophore.EquiPharm_Hungarian_v2 import screening as hungarian_v2_screening
from pharmacophore.EquiPharm_Hungarian_v3 import cli as hungarian_v3_cli
from pharmacophore.EquiPharm_Hungarian_v3 import screening as hungarian_v3_screening
from pharmacophore.EquiPharm_Hungarian_v4 import screening as hungarian_v4_screening
from pharmacophore.EquiPharm_Hungarian_v5_hard import screening as hungarian_v5_hard_screening
from pharmacophore.EquiPharm_Hungarian_v5_soft import screening as hungarian_v5_soft_screening
from pharmacophore.Equiformer_with_optimization import cli as equiformer_cli
from pharmacophore.Equiformer_with_optimization import screening as equiformer_screening
from pharmacophore.CDPKit import cli as cdpkit_cli
from pharmacophore.CDPKit import screening as cdpkit_screening
from pharmacophore.Pharmit import cli as pharmit_cli
from pharmacophore.Pharmit import screening as pharmit_screening
from pharmacophore.PharmacoMatch import cli as pharmacomatch_cli
from pharmacophore.PharmacoMatch import screening as pharmacomatch_screening
from pharmacophore import run_all_screening


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def __neg__(self):
        return FakeTensor([-value for value in self.values])

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


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
        self.assertEqual(kwargs["matching_method"], "hungarian_euclidean")
        self.assertEqual(kwargs["matching_score_mode"], "embedding_distance")
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
        self.assertEqual(kwargs["matching_method"], "hungarian_euclidean")
        self.assertEqual(kwargs["matching_score_mode"], "embedding_geometry_distance")
        self.assertEqual(kwargs["model_module"], "benchmarking.Methods.equiformer_encoder_matching")

    def test_equipharm_hungarian_3d_wrapper_sets_expected_defaults(self):
        with patch.object(hungarian_3d_screening, "screen_actives_decoys_matching") as run:
            run.return_value = {"roc_auc": 1.0}
            result = hungarian_3d_screening.run_equipharm_hungarian_3d_screening(
                checkpoint_path="checkpoint.pt",
                query_ligand="query.mol2",
                actives_dir="actives_sdf",
                decoys_dir="decoys_sdf",
                output_dir="pharmacophore/results/EquiPharm_Hungarian_3D/aces",
            )

        self.assertEqual(result, {"roc_auc": 1.0})
        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["pipeline_name"], "EquiPharm_Hungarian_3D")
        self.assertEqual(kwargs["matching_method"], "hungarian_3d")
        self.assertEqual(kwargs["matching_score_mode"], "geometry_distance")
        self.assertEqual(kwargs["model_module"], "benchmarking.Methods.equiformer_encoder_matching")

    def test_equipharm_hungarian_v3_wrapper_sets_expected_defaults(self):
        with patch.object(hungarian_v3_screening, "screen_actives_decoys_matching") as run:
            run.return_value = {"roc_auc": 1.0}
            result = hungarian_v3_screening.run_equipharm_hungarian_v3_screening(
                checkpoint_path="checkpoint.pt",
                query_ligand="query.mol2",
                actives_dir="actives_sdf",
                decoys_dir="decoys_sdf",
                output_dir="pharmacophore/results/EquiPharm_Hungarian_v3/aces",
            )

        self.assertEqual(result, {"roc_auc": 1.0})
        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["pipeline_name"], "EquiPharm_Hungarian_v3")
        self.assertEqual(kwargs["matching_method"], "hungarian_euclidean")
        self.assertEqual(kwargs["matching_score_mode"], "geometry_distance")
        self.assertEqual(kwargs["model_module"], "benchmarking.Methods.equiformer_encoder_matching")

    def test_equipharm_hungarian_v4_wrapper_sets_expected_defaults(self):
        with patch.object(hungarian_v4_screening, "screen_actives_decoys_matching") as run:
            run.return_value = {"roc_auc": 1.0}
            result = hungarian_v4_screening.run_equipharm_hungarian_v4_screening(
                checkpoint_path="checkpoint.pt",
                query_ligand="query.mol2",
                actives_dir="actives_sdf",
                decoys_dir="decoys_sdf",
                output_dir="pharmacophore/results/EquiPharm_Hungarian_v4/aces",
            )

        self.assertEqual(result, {"roc_auc": 1.0})
        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["pipeline_name"], "EquiPharm_Hungarian_v4")
        self.assertEqual(kwargs["matching_method"], "hungarian_gaussian")
        self.assertEqual(kwargs["matching_score_mode"], "tiered_distance_geometry")
        self.assertEqual(kwargs["distance_sigma"], 1.0)
        self.assertEqual(kwargs["geometry_penalty_weight"], 1.0)
        self.assertTrue(kwargs["enforce_feature_family"])

    def test_equipharm_hungarian_v5_wrappers_differ_only_in_coverage_policy(self):
        common = {
            "checkpoint_path": "checkpoint.pt",
            "query_ligand": "query.mol2",
            "actives_dir": "actives_sdf",
            "decoys_dir": "decoys_sdf",
            "output_dir": "results",
        }
        with patch.object(hungarian_v5_soft_screening, "screen_actives_decoys_matching") as soft_run:
            hungarian_v5_soft_screening.run_equipharm_hungarian_v5_soft_screening(**common)
        with patch.object(hungarian_v5_hard_screening, "screen_actives_decoys_matching") as hard_run:
            hungarian_v5_hard_screening.run_equipharm_hungarian_v5_hard_screening(**common)

        soft = soft_run.call_args.kwargs
        hard = hard_run.call_args.kwargs
        self.assertEqual(soft["matching_method"], "hungarian_cosine_quality")
        self.assertEqual(soft["matching_score_mode"], "hybrid_local_geometry")
        self.assertEqual(soft["embedding_weight"], 0.4)
        self.assertEqual(soft["spatial_weight"], 0.6)
        self.assertEqual(soft["geometry_penalty_weight"], 0.3)
        self.assertFalse(soft["require_full_query_coverage"])
        self.assertTrue(hard["require_full_query_coverage"])

    def test_equipharm_hungarian_cosine_wrapper_sets_expected_defaults(self):
        with patch.object(hungarian_cosine_screening, "screen_actives_decoys_matching") as run:
            run.return_value = {"roc_auc": 1.0}
            result = hungarian_cosine_screening.run_equipharm_hungarian_cosine_screening(
                checkpoint_path="checkpoint.pt",
                query_ligand="query.mol2",
                actives_dir="actives_sdf",
                decoys_dir="decoys_sdf",
                output_dir="pharmacophore/results/EquiPharm_Hungarian_Cosine/aces",
            )

        self.assertEqual(result, {"roc_auc": 1.0})
        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["pipeline_name"], "EquiPharm_Hungarian_Cosine")
        self.assertEqual(kwargs["matching_method"], "hungarian")
        self.assertEqual(kwargs["matching_score_mode"], "cosine")
        self.assertEqual(kwargs["model_module"], "benchmarking.Methods.equiformer_encoder_matching")

    def test_equipharm_hungarian_cosine_v2_wrapper_sets_expected_defaults(self):
        with patch.object(hungarian_cosine_v2_screening, "screen_actives_decoys_matching") as run:
            run.return_value = {"roc_auc": 1.0}
            result = hungarian_cosine_v2_screening.run_equipharm_hungarian_cosine_v2_screening(
                checkpoint_path="checkpoint.pt",
                query_ligand="query.mol2",
                actives_dir="actives_sdf",
                decoys_dir="decoys_sdf",
                output_dir="pharmacophore/results/EquiPharm_Hungarian_Cosine_v2/aces",
            )

        self.assertEqual(result, {"roc_auc": 1.0})
        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["pipeline_name"], "EquiPharm_Hungarian_Cosine_v2")
        self.assertEqual(kwargs["matching_method"], "hungarian")
        self.assertEqual(kwargs["matching_score_mode"], "cosine_geometry")
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

    def test_all_runner_can_exclude_pipelines(self):
        args = types.SimpleNamespace(exclude_pipeline=["EquiPharm_Hungarian", "EquiPharm_Hungarian_v2"])

        pipelines = run_all_screening.selected_pipelines(args)

        self.assertIn("EquiPharm", pipelines)
        self.assertNotIn("EquiPharm_Hungarian", pipelines)
        self.assertNotIn("EquiPharm_Hungarian_v2", pipelines)

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

    def test_cdpkit_wrapper_runs_cdpl_alignment_and_pools_scores(self):
        candidates = [(Path("active_0001.sdf"), 1), (Path("decoy_0001.sdf"), 0)]
        with (
            patch.object(cdpkit_screening, "collect_labeled_sdf_files", return_value=candidates),
            patch.object(cdpkit_screening, "shutil") as shutil_mock,
            patch.object(cdpkit_screening, "write_sdf_bundle") as write_bundle,
            patch.object(cdpkit_screening, "resolve_psdcreate", return_value=Path("psdcreate")),
            patch.object(cdpkit_screening, "create_psd_database") as create_psd,
            patch.object(
                cdpkit_screening,
                "align_psd_to_query",
                side_effect=[
                    [{"score": 2.5, "mol_idx": 0, "conf_idx": 0}],
                    [{"score": 0.25, "mol_idx": 0, "conf_idx": 0}],
                ],
            ) as align,
            patch.object(cdpkit_screening, "write_baseline_outputs", return_value={"roc_auc": 1.0}) as write_outputs,
        ):
            result = cdpkit_screening.run_cdpkit_screening(
                query_pharmacophore="query.pml",
                actives_dir="actives_sdf",
                decoys_dir="decoys_sdf",
                output_dir="pharmacophore/results/CDPKit/aces",
                target_name="aces",
            )

        self.assertEqual(result, {"roc_auc": 1.0})
        self.assertTrue(shutil_mock.copyfile.called)
        self.assertEqual(write_bundle.call_count, 2)
        self.assertEqual(create_psd.call_count, 2)
        self.assertEqual(align.call_count, 2)
        rows = write_outputs.call_args.args[1]
        self.assertEqual(rows[0]["pipeline"], "CDPKit")
        self.assertAlmostEqual(rows[0]["score"], 2.5, places=9)
        self.assertAlmostEqual(rows[1]["score"], 0.25, places=9)

    def test_cdpkit_functional_batch_api_returns_scores_by_file_stem(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir) / "work"
            first = Path(tmpdir) / "mol_a.sdf"
            second = Path(tmpdir) / "mol_b.sdf"
            first.write_text("a\n$$$$\n", encoding="utf-8")
            second.write_text("b\n$$$$\n", encoding="utf-8")

            with (
                patch.object(cdpkit_screening, "write_sdf_bundle") as write_bundle,
                patch.object(cdpkit_screening, "resolve_psdcreate", return_value=None),
                patch.object(cdpkit_screening, "create_psd_database") as create_psd,
                patch.object(
                    cdpkit_screening,
                    "align_psd_to_query",
                    return_value=[
                        {"score": 1.0, "mol_idx": 0, "conf_idx": 0},
                        {"score": 2.0, "mol_idx": 0, "conf_idx": 1},
                        {"score": 0.5, "mol_idx": 1, "conf_idx": 0},
                    ],
                ) as align,
            ):
                scores = cdpkit_screening.score_cdpkit_alignment_batch(
                    query_pharmacophore="query.pml",
                    candidate_sdfs=[first, second],
                    work_dir=work_dir,
                    psdcreate_bin=None,
                )

        self.assertEqual(scores, {"mol_a": 2.0, "mol_b": 0.5})
        self.assertTrue(write_bundle.called)
        self.assertTrue(create_psd.called)
        self.assertTrue(align.called)

    def test_cdpkit_functional_single_api_returns_one_score(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            candidate = Path(tmpdir) / "mol_a.sdf"
            candidate.write_text("a\n$$$$\n", encoding="utf-8")

            with patch.object(cdpkit_screening, "score_cdpkit_alignment_batch", return_value={"mol_a": 3.25}) as batch:
                score = cdpkit_screening.score_cdpkit_alignment(
                    query_pharmacophore="query.pml",
                    candidate_sdf=candidate,
                    work_dir=Path(tmpdir) / "work",
                )

        self.assertEqual(score, 3.25)
        self.assertEqual(batch.call_args.kwargs["candidate_sdfs"], [candidate])

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

    def test_pharmacomatch_wrapper_fails_when_all_candidates_fail(self):
        candidates = [(Path("active_0001.sdf"), 1)]
        with (
            patch.object(pharmacomatch_screening, "collect_labeled_sdf_files", return_value=candidates),
            patch.object(pharmacomatch_screening, "run_command", side_effect=FileNotFoundError("pharmacomatch")),
        ):
            with self.assertRaisesRegex(RuntimeError, "produced no valid scores"):
                pharmacomatch_screening.run_pharmacomatch_screening(
                    command_template="pharmacomatch --query {query_ligand} --candidate {candidate}",
                    score_json_key="score",
                    query_ligand="query.mol2",
                    actives_dir="actives_sdf",
                    decoys_dir="decoys_sdf",
                    output_dir="pharmacophore/results/PharmacoMatch/aces",
                    target_name="aces",
                )

    def test_official_pharmacomatch_rows_convert_penalty_to_descending_score(self):
        import pandas as pd

        screener = Mock()
        screener.active_ligand_score = FakeTensor([3.0, 1.5])
        screener.inactive_ligand_score = FakeTensor([5.0])
        metadata = Mock()
        metadata.active = pd.DataFrame({"name": ["active_a", "active_a", "active_b"]})
        metadata.inactive = pd.DataFrame({"name": ["decoy_a"]})

        rows = pharmacomatch_screening._rows_from_official_screener(
            screener=screener,
            metadata=metadata,
            target_name="aces",
            pipeline_name="PharmacoMatch",
            vs_path=Path("external/PharmacoMatch/data/DUD-E/ACES"),
        )

        self.assertEqual([row["name"] for row in rows], ["active_a", "active_b", "decoy_a"])
        self.assertEqual([row["label"] for row in rows], [1, 1, 0])
        self.assertEqual([row["score"] for row in rows], [-3.0, -1.5, -5.0])

    def test_official_pharmacomatch_missing_inputs_are_clear(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(FileNotFoundError, "Official PharmacoMatch inputs are missing"):
                pharmacomatch_screening._validate_official_pharmacomatch_inputs(
                    Path(tmpdir) / "ACES",
                    Path(tmpdir) / "trained_model.ckpt",
                )

    def test_prepare_official_pharmacomatch_target_builds_raw_inputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "data" / "DUD-E" / "aces"
            (target / "actives_sdf").mkdir(parents=True)
            (target / "decoys_sdf").mkdir()
            (target / "query.pml").write_text("query\n", encoding="utf-8")
            (target / "actives_sdf" / "a.sdf").write_text("active\n$$$$\n", encoding="utf-8")
            (target / "decoys_sdf" / "d.sdf").write_text("decoy\n$$$$\n", encoding="utf-8")
            cdpkit_bin = root / "CDPKit" / "Bin"
            cdpkit_bin.mkdir(parents=True)
            (cdpkit_bin / "psdcreate").write_text("", encoding="utf-8")

            with patch.object(pharmacomatch_screening, "run_command") as run_command:
                prepared = pharmacomatch_screening.prepare_official_pharmacomatch_target(
                    target_dir=target,
                    prepared_vs_dir=root / "prepared" / "aces",
                    cdpkit_bin=cdpkit_bin,
                )

            self.assertEqual(prepared, root / "prepared" / "aces")
            self.assertTrue((prepared / "raw" / "query.pml").exists())
            self.assertTrue((prepared / "preprocessing" / "actives.sdf").exists())
            self.assertTrue((prepared / "preprocessing" / "inactives.sdf").exists())
            self.assertEqual(run_command.call_count, 2)
            self.assertEqual(run_command.call_args_list[0].args[0][0], str(cdpkit_bin / "psdcreate"))

    def test_prepare_official_pharmacomatch_target_falls_back_to_python_cdpl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "data" / "DUD-E" / "aces"
            (target / "actives_sdf").mkdir(parents=True)
            (target / "decoys_sdf").mkdir()
            (target / "query.pml").write_text("query\n", encoding="utf-8")
            (target / "actives_sdf" / "a.sdf").write_text("active\n$$$$\n", encoding="utf-8")
            (target / "decoys_sdf" / "d.sdf").write_text("decoy\n$$$$\n", encoding="utf-8")

            with (
                patch.object(pharmacomatch_screening, "_resolve_psdcreate", return_value=None),
                patch.object(pharmacomatch_screening, "_create_psd_with_cdpl_python") as create_psd,
            ):
                prepared = pharmacomatch_screening.prepare_official_pharmacomatch_target(
                    target_dir=target,
                    prepared_vs_dir=root / "prepared" / "aces",
                )

            self.assertEqual(prepared, root / "prepared" / "aces")
            self.assertEqual(create_psd.call_count, 2)
            self.assertEqual(create_psd.call_args_list[0].args[1], prepared / "raw" / "actives.psd")

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
            "query_pharmacophore": "data/CDPKit/aces/query.pml",
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

    def test_cdpkit_cli_generates_query_from_target_dir_when_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target_dir = Path(tmpdir) / "aces"
            (target_dir / "actives_sdf").mkdir(parents=True)
            (target_dir / "decoys_sdf").mkdir()
            (target_dir / "crystal_ligand.mol2").write_text("@<TRIPOS>MOLECULE\n", encoding="utf-8")
            generated_query = target_dir / "query.pml"
            argv = [
                "cli",
                "--target-dir",
                str(target_dir),
                "--output-dir",
                "pharmacophore/results/CDPKit/aces",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(sys, "stdout", StringIO()),
                patch.object(cdpkit_cli, "ensure_cdpkit_query", return_value=generated_query) as ensure_query,
                patch.object(cdpkit_cli, "run_cdpkit_screening") as run,
            ):
                run.return_value = {"pipeline": "CDPKit", "target": "aces"}
                cdpkit_cli.main()

        self.assertEqual(ensure_query.call_args.args[0], target_dir)
        self.assertEqual(run.call_args.kwargs["query_pharmacophore"], str(generated_query))

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
