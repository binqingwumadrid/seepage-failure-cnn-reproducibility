# Inverse prediction (physical-field image to parameters)

This directory packages the archived inverse CNN as portable command-line
code. It predicts cohesion `c`, friction angle `phi`, and `log10(ks)` from an
RGB pseudocolor field image. No old experiment files are modified.

## Default configuration

- input: equivalent-plastic-strain images with 3% Gaussian noise;
- split: 900 training images and 100 test images;
- resize: 256 x 512 RGB;
- model: four stride-2 convolution blocks with GroupNorm, followed by
  fully connected layers `256*16*32 -> 512 -> 128 -> 3`;
- optimizer: Adam, learning rate `1e-4`, batch size `4`;
- training: 200 epochs; validation at epochs 1, 50, 100, 150, and 200;
- loss: MSE on the three normalized parameters.

The default parameter bounds follow Table 1 of the manuscript:
`c=3000..7000 Pa`, `phi=18..27 degrees`, and `log10(ks)=-2.3..-0.3`.
The archived 1000-case grid reaches 6600 Pa for cohesion, while 7000 Pa remains
the nominal upper domain bound used consistently by both learning directions.

Each run creates a fresh random seed unless `--seed` is supplied. The actual
seed and exact split are saved in `run_config.json` and `data_split.csv`.

## Environment

Use the isolated interpreter at the repository root; Anaconda `base` is not changed:

```powershell
.\.venv\python.exe .\inverse_prediction\smoke_test.py
```

## Optional full training

Full training is not required for the GitHub upload. If a reader wants to run
it, the defaults resolve the data directory automatically:

```powershell
.\.venv\python.exe .\inverse_prediction\train_inverse.py
```

Equivalent explicit command:

```powershell
.\.venv\python.exe .\inverse_prediction\train_inverse.py `
  --data-dir .\generated_data\equivalent_plastic_strain\noise_3pct `
  --output-dir .\inverse_prediction\outputs\equivalent_plastic_strain_noise_3pct `
  --train-size 900 --test-size 100 --epochs 200 --batch-size 4
```

Omit `--seed` for different results on every run. To reproduce a particular
run, copy the recorded seed from `run_config.json` and pass `--seed NUMBER`.

## Predict one image from a trained checkpoint

```powershell
.\.venv\python.exe .\inverse_prediction\predict_inverse.py `
  --checkpoint .\inverse_prediction\outputs\equivalent_plastic_strain_noise_3pct\best_model.pt `
  --image .\generated_data\equivalent_plastic_strain\noise_3pct\c3000_phi18.00_logk-2.300000.png `
  --output .\inverse_prediction\example_prediction.json
```

Training results, checkpoints, and the local virtual environment are excluded
by `.gitignore`. The repository only needs the source code, raw data, and
documentation; generated model results do not need to be uploaded.
