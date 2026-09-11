# Data description

## Raw SPH snapshots

The main dataset contains 1000 final-step particle snapshots:

```text
raw_sph_data/
  dabashenliu_run_001/particle_positions_150000.csv
  ...
  dabashenliu_run_1000/particle_positions_150000.csv
```

Only the final snapshot is required by the published image-processing and
learning workflow. Earlier time-step files are therefore not included.

The image generator uses these CSV columns:

| Column | Meaning | Unit/use |
| --- | --- | --- |
| `x`, `y` | Particle coordinates | Model length unit |
| `id` | Material/particle identifier | Soil particles use ID 1 |
| `equivalent_plastic_strain` | Equivalent plastic strain | Dimensionless |
| `pressure` | Pore-water pressure | Pa |
| `cohesion_0` | Initial cohesion | Pa |
| `frictional_angle` | Internal friction angle | Degree |
| `ks` | Permeability coefficient | m/s |

## Parameter ranges

The nominal manuscript domain is `c=3000..7000 Pa`, `phi=18..27 degrees`, and
`ks=0.005..0.5 m/s`. The archived 1000-case grid contains cohesion values from
3000 through 6600 Pa; 7000 Pa remains the nominal upper normalization bound
reported in Table 1. Parameter labels are read directly from each CSV rather
than inferred from the run number.

## Derived images

`generate_pseudocolor_images.m` reconstructs the image datasets using natural
neighbor interpolation, Gaussian smoothing, fixed physical value ranges, and
the MATLAB `jet(256)` color map. It writes `generation_manifest.csv`, which
records every source CSV, output name, field, parameter value, noise setting,
and generation status.

Generated PNG files are deliberately ignored by Git. They are intermediate
data and are recreated locally before training. The manifests in
`generated_data/equivalent_plastic_strain/` and `generated_data/pressure/`
document the previously verified 1000-case generation.

## Modified-geometry SPH snapshots

The separate modified-geometry dataset contains 1000 parameter combinations:

```text
raw_sph_data_modified_geometry/
  dabashenliu_run_0001/
    particle_positions_000001.csv
    particle_positions_150000.csv
  ...
  dabashenliu_run_1000/
    particle_positions_000001.csv
    particle_positions_150000.csv
```

Its grid contains 10 values for each parameter: `c=3000..7000 Pa`,
`phi=18..28 degrees`, and `log10(ks)=-2.3..-0.3`, giving 1000 unique
combinations. Parameter labels are stored in the CSV columns and have been
checked against the run-number mapping.

Machine-specific SPH configuration JSON files are retained locally but are
excluded from the release. The independent SPH solver and its private geometry
inputs are not distributed.
