# Forward prediction (parameters to physical-field image)

This directory implements the forward CNN using the archived experiment code
as the primary reference. The input is
`[cohesion c, friction angle phi, log10(permeability ks)]`; the output is a
three-channel RGB pseudocolor image. The default task is the paper's main
equivalent-plastic-strain experiment. The same entry point can also train the
pore-water-pressure extension.

## Default method

- random split: 900 training images and 100 test images;
- three fully connected layers followed by three transposed convolutions;
- sigmoid RGB output;
- Adam optimizer, learning rate `1e-3`, batch size `4`, and `150` epochs;
- loss: `MSE + 0.2 * (1 - SSIM) + 0.1 * gradient_loss`;
- a fresh random seed per run, recorded together with the exported `data_split.csv`;
- MSE, MAE, SSIM, gradient loss, per-image R2, and global pixel R2;
- warm-up and repeated model-only inference timing.

The resized images are cached in RAM and validation runs at epochs 1, 50, 100,
and 150 to reduce repeated image loading and validation overhead. The random
split, parameter initialization, and loader order use the recorded seed. Fast
cuDNN kernels remain enabled because forcing deterministic transposed
convolutions causes a severe slowdown on the RTX 5060; the setting is recorded
in `run_config.json`.

The old experimental script directly fed parameters with very different
scales (`c` is about 3000--7000, while `log10(ks)` is about -2.3---0.3).
This version scales all three inputs to `[-1, 1]` using the physical domain
reported in the paper. It also reads `log10(ks)` directly from the corrected
image names.

## Environment (Windows, NVIDIA GPU)

Create an isolated `.venv` at the repository root; this does not modify an
existing Anaconda `base` environment. From Anaconda Prompt, run:

```powershell
conda create --prefix .\.venv python=3.11 pip -y
.\.venv\python.exe -m pip install torch==2.11.0 `
  --index-url https://download.pytorch.org/whl/cu128
.\.venv\python.exe -m pip install -r .\forward_prediction\requirements.txt
.\.venv\python.exe .\forward_prediction\smoke_test.py
```

The tested versions are Python 3.11.16, PyTorch 2.11.0+cu128, NumPy 2.4.6,
and Pillow 12.3.0. Confirm the GPU connection with:

```powershell
.\.venv\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

In VS Code, use **Python: Select Interpreter** and select
`.venv\python.exe` for this repository only.

## Train the main equivalent-plastic-strain model

The defaults below already point to the sibling dataset from the repository
root:

```powershell
.\.venv\python.exe .\forward_prediction\train_forward.py
```

Equivalent explicit command:

```powershell
.\.venv\python.exe .\forward_prediction\train_forward.py `
  --field equivalent_plastic_strain `
  --data-dir .\generated_data\equivalent_plastic_strain\noise_0pct `
  --output-dir .\forward_prediction\outputs\equivalent_plastic_strain `
  --train-size 900 --test-size 100 --epochs 150 --batch-size 4 `
  --image-height 128 --image-width 256
```

Omitting `--seed` intentionally creates a different split and initialization
for every run. To reproduce one completed run exactly, read its seed from
`run_config.json` and pass it explicitly, for example `--seed 173829104`.

To retrain from scratch in the same output location, add
`--overwrite-output`. Existing results are otherwise protected.

## Train the pressure-field extension

```powershell
.\.venv\python.exe .\forward_prediction\train_forward.py `
  --field pressure `
  --data-dir .\generated_data\pressure\noise_0pct `
  --output-dir .\forward_prediction\outputs\pressure `
  --train-size 900 --test-size 100 --epochs 150 --batch-size 4 `
  --image-height 128 --image-width 256
```

Both source image sets are resized deterministically to 128 x 256 pixels at
load time, matching the archived complete Python experiment.

## Predict one new case

```powershell
.\.venv\python.exe .\forward_prediction\predict_forward.py `
  --checkpoint .\forward_prediction\outputs\equivalent_plastic_strain\best_model.pt `
  --cohesion-pa 4600 --friction-angle-deg 24 --ks 0.079 `
  --output .\forward_prediction\example_prediction.png
```

The command saves both the predicted PNG and a JSON file containing its input
parameters and checkpoint provenance. Values outside the paper's parameter
domain are rejected unless `--allow-extrapolation` is explicitly supplied.

## Outputs

Each completed run contains:

- `run_config.json`: complete settings, software/device information, and model size;
- `data_split.csv`: exact reproducible training/test membership;
- `training_history.csv`: every epoch's training metrics and scheduled test metrics;
- `best_model.pt` and `last_model.pt`: checkpoints with embedded configuration;
- `test_sample_metrics.csv`: metrics for every test image;
- `summary.json`: aggregate metrics and training/inference timing;
- `comparisons/`: ground truth on the left and prediction on the right.

The reported final test metrics use `last_model.pt` at epoch 150, matching the
archived complete code. `best_model.pt` is also retained from the scheduled
validation epochs for applications that prefer the lowest observed test loss.

## Paper/code resolution note

The paper text states a latent tensor of `256 x 8 x 16`; after three 2x
upsampling layers that produces a 64 x 128 image. The archived complete code,
however, trains at 128 x 256 and therefore uses `256 x 16 x 32`. The default
here follows the archived experiment. To follow the literal paper dimensions,
pass `--image-height 64 --image-width 128`. The selected dimensions and latent
shape are always recorded in `run_config.json`.

## Verification performed on the target computer

The smoke test passed with CUDA enabled on the NVIDIA GeForce RTX 5060 Laptop
GPU. A real four-image optimization step at 128 x 256 also passed: input shape
`(4, 3)`, output shape `(4, 3, 128, 256)`, 68,032,067 trainable parameters,
and about 1.38 GiB peak reserved GPU memory during that check.
