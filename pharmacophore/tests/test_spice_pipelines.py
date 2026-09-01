"""Dependency-light contract tests for the isolated SPICE screening family."""

import ast
import json
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
    assert source.count("kwargs.update(SPICE_MODEL)") == 2


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
