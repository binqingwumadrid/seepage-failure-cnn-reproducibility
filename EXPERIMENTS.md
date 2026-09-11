# Experiment commands and outputs

Run commands from the repository root. Full training is optional for code
review; the smoke tests verify model construction and optimization without
creating publication results.

## Image construction

Open MATLAB, add `image_generation` to the path, and follow
`image_generation/README.md`. The same generator supports equivalent plastic
strain, pressure, arbitrary listed noise levels, and alternate resolutions.

## Main forward experiment

```powershell
.\.venv\python.exe .\forward_prediction\train_forward.py `
  --data-dir .\generated_data\equivalent_plastic_strain\noise_0pct `
  --train-size 900 --test-size 100 --epochs 150
```

The run exports the loss history, per-sample image metrics, timing summary,
comparison images, exact split, and checkpoints into `forward_prediction/outputs/`.

## Main inverse experiment

```powershell
.\.venv\python.exe .\inverse_prediction\train_inverse.py `
  --data-dir .\generated_data\equivalent_plastic_strain\noise_3pct `
  --train-size 900 --test-size 100 --epochs 200
```

The run exports normalized and physical-scale parameter errors, per-sample
predictions, timing, exact split, and checkpoints into
`inverse_prediction/outputs/`.

## Limited-training-data experiments

The scripts accept a subset of the available 1000 images. Samples not assigned
to training or testing are marked `unused` in `data_split.csv`. For a fixed
250-image test set, run the desired training sizes with the same explicit seed:

```powershell
.\.venv\python.exe .\inverse_prediction\train_inverse.py --train-size 50  --test-size 250 --seed 20 --output-dir .\inverse_prediction\outputs\train_050
.\.venv\python.exe .\inverse_prediction\train_inverse.py --train-size 250 --test-size 250 --seed 20 --output-dir .\inverse_prediction\outputs\train_250
.\.venv\python.exe .\inverse_prediction\train_inverse.py --train-size 500 --test-size 250 --seed 20 --output-dir .\inverse_prediction\outputs\train_500
.\.venv\python.exe .\inverse_prediction\train_inverse.py --train-size 750 --test-size 250 --seed 20 --output-dir .\inverse_prediction\outputs\train_750
```

An omitted `--seed` deliberately produces a new split and initialization; an
explicit seed is required when several runs must share exactly the same test
membership.

## Noise and resolution sensitivity

Generate additional noise levels by passing, for example,
`'NoisePercent', [0 1 3 5 10 20 50]`. Generate resolution-specific datasets
with `'Nx', 400, 'Ny', 150`, `'Nx', 800, 'Ny', 300`, or
`'Nx', 1600, 'Ny', 600`, using a different `OutputDir` for each set. Point
`train_inverse.py --data-dir` to the required generated folder.

## Pressure-field extension

```powershell
.\.venv\python.exe .\forward_prediction\train_forward.py `
  --field pressure `
  --data-dir .\generated_data\pressure\noise_0pct `
  --output-dir .\forward_prediction\outputs\pressure
```

## Runtime reporting

Both training scripts measure pure optimization time and model-only inference
time after GPU warm-up. The values are written to each run's `summary.json`.
This separates one-time offline training cost from online prediction cost.
