# SPICE EquiPharm screening

This package mirrors the pooled EquiPharm pipeline and every Hungarian variant,
but loads `SPICEPharmacophoreEncoder` from
`pharm_training/equiformer_encoder_pharmaco_feat.py`.

Use a winning search trial's `checkpoints/best.pt` when possible. That checkpoint
contains both the transferable geometric core and the learned SPICE element
embedding used to initialize the 11-to-hidden descriptor projection. A newer
`trained_encoder.pt` produced by this repository also contains that embedding.

Example:

```bash
python -m pharmacophore.pharmacophore_spice.EquiPharm.cli \
  --checkpoint runs/pharm_training/spice_search/trial_ID/checkpoints/best.pt \
  --target-dir /path/to/dude/target \
  --output-dir runs/pharmacophore_spice/target \
  --device cuda
```

Replace `EquiPharm` with any included Hungarian directory to run that scoring
variant. The adapter expects the standard 11-channel atom descriptors produced
by `pharmacophore.core.molecule_io`.

