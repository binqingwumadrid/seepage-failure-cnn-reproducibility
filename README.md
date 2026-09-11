# Forward-inverse CNN framework for seepage-failure prediction

This repository contains the data and source code for the main reproducible
workflow accompanying *A Forward-Inverse Convolutional Neural Network
Framework for Accelerated Seepage Failure Prediction in Porous Geomaterials*.
It converts final-step Smoothed Particle Hydrodynamics (SPH) particle fields
to RGB pseudocolor images, predicts field images from material parameters, and
inverts field images back to the material parameters.

## Repository layout

```text
raw_sph_data/           Final SPH particle snapshot for each of 1000 cases
raw_sph_data_modified_geometry/
                        Initial and final snapshots for 1000 modified-geometry cases
image_generation/       MATLAB pseudocolor-image generator
generated_data/         Generation manifests; PNG files are regenerated locally
forward_prediction/     c, phi, log10(ks) -> RGB field image
inverse_prediction/     RGB field image -> c, phi, log10(ks)
DATA.md                  Data layout, columns, units, and parameter ranges
EXPERIMENTS.md           Commands corresponding to the reported experiments
```

The modified-geometry dataset is included as a separate 1000-case dataset.
It is not required by the main fixed-geometry image-generation and learning
workflow.

## Parameter domain

The nominal domain follows Table 1 of the manuscript:

- cohesion `c`: 3000 to 7000 Pa;
- friction angle `phi`: 18 to 27 degrees;
- permeability `ks`: 0.005 to 0.5 m/s.

The learning code uses `log10(ks)`, with practical normalization bounds of
`-2.3` to `-0.3`. Both the forward and inverse models use the same bounds.

## Environment

The verified Windows environment uses Python 3.11, PyTorch 2.11.0 with CUDA
12.8, NumPy 2.4.6, Pillow 12.3.0, and MATLAB R2016a for image generation.
Create an isolated environment in the repository; do not install into an
existing Anaconda `base` environment:

```powershell
conda create --prefix .\.venv python=3.11 pip -y
.\.venv\python.exe -m pip install torch==2.11.0 `
  --index-url https://download.pytorch.org/whl/cu128
.\.venv\python.exe -m pip install -r .\forward_prediction\requirements.txt
```

The `.venv` directory is ignored by Git.

## Reproduction workflow

1. Download the Git LFS data after cloning:

   ```powershell
   git lfs pull
   ```

2. Generate the pseudocolor images in MATLAB. See
   [`image_generation/README.md`](image_generation/README.md).

3. Verify the Python models without full training:

   ```powershell
   .\.venv\python.exe .\forward_prediction\smoke_test.py
   .\.venv\python.exe .\inverse_prediction\smoke_test.py
   ```

4. Run the desired training or sensitivity experiment using
   [`EXPERIMENTS.md`](EXPERIMENTS.md).

Training outputs and model checkpoints are intentionally not versioned. Each
training run records its actual random seed and exact train/test/unused split,
so a completed run can be repeated by passing the recorded seed explicitly.

## Data and large-file handling

The 1000 main final-step SPH CSV files and the 2000 modified-geometry initial-
and final-step CSV files are tracked with Git Large File Storage (LFS).
Generated PNG files are excluded because they can be recreated exactly from
the raw snapshots and the image-generation script. The compact generation
manifests remain versioned to document the source-to-image mapping. Local SPH
solver configurations are not released because the independent SPH solver and
its geometry inputs are outside the scope of this repository.

## License

The source code is released under the [MIT License](LICENSE). The accompanying
article and third-party methods retain their respective rights.

GitHub-compatible citation metadata is provided in [`CITATION.cff`](CITATION.cff).
The article DOI can be added after publication.
