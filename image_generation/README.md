# Pseudocolor image generation

This folder contains the MATLAB R2016a-compatible preprocessing step for
converting the final SPH particle snapshot into RGB pseudocolor images.

## Input layout

The raw-data directory must contain:

```text
dabashenliu_run_001/particle_positions_150000.csv
dabashenliu_run_002/particle_positions_150000.csv
...
dabashenliu_run_1000/particle_positions_150000.csv
```

The generator reads `cohesion_0`, `frictional_angle`, and `ks` directly
from each CSV. It does not infer physical parameters from the run number.

## Run in MATLAB

Add this folder to the MATLAB path, then run one of the following commands.
The commands below locate the repository from the generator itself and do not
contain machine-specific absolute paths:

```matlab
generator_dir = fileparts(which('generate_pseudocolor_images'));
repo_root = fileparts(generator_dir);
raw_dir = fullfile(repo_root, 'raw_sph_data');
```

Pressure field without artificial noise:

```matlab
output_dir = fullfile(repo_root, 'generated_data', 'pressure');
generate_pseudocolor_images(raw_dir, 'pressure', 'OutputDir', output_dir);
```

Equivalent plastic strain without artificial noise:

```matlab
output_dir = fullfile(repo_root, 'generated_data', 'equivalent_plastic_strain');
generate_pseudocolor_images(raw_dir, 'equivalent_plastic_strain', ...
    'OutputDir', output_dir);
```

Equivalent plastic strain with the noise levels used in a sensitivity test:

```matlab
output_dir = fullfile(repo_root, 'generated_data', 'equivalent_plastic_strain');
generate_pseudocolor_images(raw_dir, 'equivalent_plastic_strain', ...
    'NoisePercent', [0 1 3 5], 'OutputDir', output_dir);
```

For a one-case smoke test:

```matlab
test_output = fullfile(pwd, 'image_generation_smoke_test');
generate_pseudocolor_images(raw_dir, 'pressure', ...
    'RunIds', 1, 'OutputDir', test_output, 'Overwrite', true);
```

## Defaults that must match the manuscript

- Final snapshot: `particle_positions_150000.csv`
- Selected material ID: `1`
- Physical domain: x = 3 to 18, y = 0 to 6
- Pressure image size: 800 by 300 pixels
- Equivalent-plastic-strain image size: 1600 by 600 pixels
- Gaussian smoothing sigma: 2 pixels
- Pressure normalization: 0 to 30000
- Equivalent plastic strain normalization: 0 to 0.3
- Color map: MATLAB `jet(256)`
- Invalid area: white
- Vertical orientation: flipped after interpolation

Every run writes `generation_manifest.csv`, which records the source CSV,
exact physical parameters, output file, normalization, noise level, random
seed, particle count, and the number of clipped pixels.

## Important data-label correction

The raw CSV files show that the permeability values are distributed from
approximately `log10(ks) = -2.3` to `-0.3`. The earlier run-number formula
assigned some images incorrect permeability labels. The generator therefore
uses `log10(ks)` from the CSV itself and records its full value in the
manifest.
