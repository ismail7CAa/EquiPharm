"""Dependency-light contracts for SPICE-to-QM9 fine-tuning."""

import ast
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "finetune_spice_qm9.py"


def test_finetuning_script_parses_and_is_restricted_to_nine_targets():
    source = SCRIPT.read_text()
    ast.parse(source)
    assert 'TARGET_PRESET = "electronic_geometry"' in source
    assert 'DEFAULT_LEARNING_RATE = 1e-6' in source
    assert 'default=DEFAULT_LEARNING_RATE' in source
    assert 'at most 1e-5' in source


def test_finetuning_restores_both_spice_core_and_species_embedding():
    source = SCRIPT.read_text()
    assert 'payload["encoder_state_dict"]' in source
    assert 'payload.get("species_embedding_state_dict")' in source
    assert '"encoder.species_embedding.weight"' in source
    assert 'self.encoder.encode_embedded_nodes(data, embedded)' in source
