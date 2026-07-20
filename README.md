# `cap-zscore-mwe`

Minimal reproducible example for the idea that a CAP / anti-CAP pair can emerge from a single positively activated network after voxelwise z-scoring.

## Run

```bash
uv run simulate_cap_zscore_effect.py
```

Outputs go to `results/`:

- `simulation_cap_maps.png`
- `simulation_timecourses.png`
- `simulated_recording_raw.nii.gz`
- `simulated_recording_cleaned.nii.gz`
- `simulated_cap_reconstruction.nii.gz`

## What it does

- simulates a `50x50xtime` recording with one central `10x10` square
- injects stochastic positive-only activity into that square
- keeps explicit off-periods
- applies voxelwise temporal z-scoring
- then applies per-frame spatial z-scoring + L2 normalization
- fits `k=2` CAPs with spherical k-means in that correlation-like space
- compares CAP labels against:
  - activation present
  - latent amplitude above/below its mean

## Take-home message

The positive CAP tends to track `latent amplitude > mean` better than `activation present`, which is the effect this MWE is meant to illustrate.
