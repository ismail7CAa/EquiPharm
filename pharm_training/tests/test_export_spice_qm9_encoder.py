from pathlib import Path

import torch

from pharm_training.export_spice_qm9_encoder import export_encoder


def test_exports_finetuned_core_in_screening_format(tmp_path: Path):
    tuned = tmp_path / "best_model.pt"
    source = tmp_path / "spice.pt"
    output = tmp_path / "trained_encoder.pt"
    torch.save(
        {
            "epoch": 3,
            "model_state_dict": {
                "encoder.model.layer.weight": torch.ones(2, 2),
                "encoder.species_embedding.weight": torch.ones(5, 2),
                "regression_head.weight": torch.zeros(9, 2),
            },
        },
        tuned,
    )
    torch.save({"architecture": {"hidden_dim": 2}, "elements": [1, 6, 7, 8, 9]}, source)

    export_encoder(tuned, source, output)
    payload = torch.load(output, weights_only=False)

    assert set(payload["encoder_state_dict"]) == {"layer.weight"}
    assert "regression_head.weight" not in payload["encoder_state_dict"]
    assert payload["dataset"] == "spice_qm9_finetune"
    assert payload["qm9_epoch"] == 3
