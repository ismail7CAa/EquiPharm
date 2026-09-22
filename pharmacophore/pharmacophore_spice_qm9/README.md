# SPICE → QM9 pharmacophore screening

This isolated pipeline screens with the geometric encoder pretrained on SPICE
and subsequently fine-tuned on the nine selected QM9 targets. It intentionally
rejects ordinary SPICE and legacy QM9 checkpoints.

First export the fine-tuned core (this does not retrain the model):

```bash
python -m pharm_training.export_spice_qm9_encoder \
  --checkpoint /runs/pharm_training/spice_qm9_finetune_300/checkpoints/best_model.pt \
  --output /runs/pharm_training/spice_qm9_finetune_300/checkpoints/trained_encoder.pt
```

The source SPICE checkpoint is read from `transfer_provenance.json`. Pass
`--source-spice-checkpoint /path/to/spice/checkpoints/best.pt` only if that
record or its referenced file is unavailable.

Then screen one target with three seeds using Hungarian v2 (Euclidean
assignment followed by embedding-geometry distance scoring):

```bash
python -m pharmacophore.pharmacophore_spice_qm9.EquiPharm_Hungarian_v2.cli \
  --checkpoint /runs/pharm_training/spice_qm9_finetune_300/checkpoints/trained_encoder.pt \
  --target-dir /path/to/dude/target \
  --device cuda
```
