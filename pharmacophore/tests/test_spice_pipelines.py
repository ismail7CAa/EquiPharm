"""Dependency-light contract tests for the isolated SPICE screening family."""

import ast
from pathlib import Path


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
    assert {path.name for path in ROOT.iterdir() if path.is_dir()} == variants
    for variant in variants:
        for filename in ("__init__.py", "screening.py", "cli.py"):
            ast.parse((ROOT / variant / filename).read_text())
        assert (ROOT / variant / "configs" / "target.example.json").is_file()


def test_shared_runner_forces_the_spice_adapter():
    source = (ROOT / "common.py").read_text()
    assert '"model_module": "pharm_training.equiformer_encoder_pharmaco_feat"' in source
    assert '"model_class": "SPICEPharmacophoreEncoder"' in source
    assert source.count("kwargs.update(SPICE_MODEL)") == 2
