"""Dependency-light contract tests for the isolated SPICE screening family."""

import ast
import importlib
import json
import sys
import types
from pathlib import Path

import pytest

from pharmacophore.core.seed_aggregation import run_seeded


ROOT = Path(__file__).parents[1] / "pharmacophore_spice"


def test_all_spice_pipeline_modules_are_present_and_parse():
    variants = {
        "EquiPharm",
        "EquiPharm_Hungarian",
        "EquiPharm_Hungarian_3D",
        "EquiPharm_Hungarian_Cosine",
        "EquiPharm_Hungarian_Cosine_v2",
        "EquiPharm_Hungarian_v2",
        "EquiPharm_Hungarian_v3",
        "EquiPharm_Hungarian_v4",
        "EquiPharm_Hungarian_v5_hard",
        "EquiPharm_Hungarian_v5_soft",
    }
    assert {path.name for path in ROOT.iterdir() if path.is_dir() and not path.name.startswith("__")} == variants
    for variant in variants:
        for filename in ("__init__.py", "screening.py", "cli.py"):
            ast.parse((ROOT / variant / filename).read_text())
        assert (ROOT / variant / "configs" / "target.example.json").is_file()


def test_shared_runner_forces_the_spice_adapter():
    source = (ROOT / "common.py").read_text()
    assert '"model_module": "pharm_training.equiformer_encoder_pharmaco_feat"' in source
    assert '"model_class": "SPICEPharmacophoreEncoder"' in source
    assert source.count("for key, value in SPICE_MODEL.items()") == 2


def test_cli_defaults_to_results_directory_and_writes_log(tmp_path, monkeypatch):
    matching_module = types.ModuleType("pharmacophore.core.matching_screening")
    matching_module.screen_actives_decoys_matching = lambda **kwargs: kwargs
    screening_module = types.ModuleType("pharmacophore.core.screening")
    screening_module.screen_actives_decoys = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, matching_module.__name__, matching_module)
    monkeypatch.setitem(sys.modules, screening_module.__name__, screening_module)
    sys.modules.pop("pharmacophore.pharmacophore_spice.common", None)
    run_cli = importlib.import_module("pharmacophore.pharmacophore_spice.common").run_cli

    target_dir = tmp_path / "data" / "aces"
    checkpoint = tmp_path / "best.pt"
    captured = {}

    def runner(**kwargs):
        captured.update(kwargs)
        print("screening output")
        return {"roc_auc": 0.5}

    runner.__module__ = "pharmacophore.pharmacophore_spice.EquiPharm.screening"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["cli", "--checkpoint", str(checkpoint), "--target-dir", str(target_dir), "--device", "cpu", "--seed", "1"],
    )

    run_cli(runner, "test")

    output = tmp_path / "pharmacophore" / "results" / "pharmacophore_spice" / "EquiPharm" / "aces"
    assert Path(captured["output_dir"]) == Path("pharmacophore/results/pharmacophore_spice/EquiPharm/aces/seed_1")
    assert captured["checkpoint_path"] == str(checkpoint)
    assert "screening output" in (output / "run.log").read_text()


def test_seeded_runner_uses_three_samples_and_writes_mean(tmp_path):
    calls = []

    def runner(**kwargs):
        calls.append(kwargs)
        seed = kwargs["seed"]
        return {"roc_auc": seed / 10, "n_actives": 50, "pipeline": "test"}

    result = run_seeded(runner, {"output_dir": str(tmp_path)}, seeds=(1, 2, 3))

    assert [call["seed"] for call in calls] == [1, 2, 3]
    assert [Path(call["output_dir"]).name for call in calls] == ["seed_1", "seed_2", "seed_3"]
    assert result["mean"]["roc_auc"] == pytest.approx(0.2)
    assert result["mean"]["n_actives"] == 50
    written = json.loads((tmp_path / "seed_mean" / "metrics.json").read_text())
    assert written == result
