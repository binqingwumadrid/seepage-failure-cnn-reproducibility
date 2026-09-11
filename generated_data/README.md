# Generated pseudocolor image dataset

## Source

Each image was generated from the final SPH snapshot
`particle_positions_150000.csv` in one of 1,000 simulation directories.
Only particles with material ID 1 were used. Invalid regions outside the
interpolated particle domain are white.

The labels are read from the CSV columns `cohesion_0`,
`frictional_angle`, and `ks`. The permeability label is stored as
`log10(ks)`. Labels are not inferred from the run number.

## Contents

```text
generated_data/
├── pressure/
│   ├── generation_manifest.csv
│   └── noise_0pct/                 1,000 RGB PNG files, 800 x 300
└── equivalent_plastic_strain/
    ├── generation_manifest.csv
    ├── noise_0pct/                 1,000 RGB PNG files, 1600 x 600
    ├── noise_1pct/                 1,000 RGB PNG files, 1600 x 600
    ├── noise_3pct/                 1,000 RGB PNG files, 1600 x 600
    └── noise_5pct/                 1,000 RGB PNG files, 1600 x 600
```

There are 5,000 images in total. Every image uses MATLAB `jet(256)` as
the pseudocolor map.

## Processing settings

- Physical domain: x = 3 to 18, y = 0 to 6
- Interpolation: natural-neighbor scattered interpolation
- Spatial smoothing: Gaussian sigma of 2 pixels
- Pressure normalization range: 0 to 30,000
- Equivalent-plastic-strain normalization range: 0 to 0.3
- Values outside the normalization range are clipped to [0, 1]
- Noise is added in normalized-value space before pseudocolor mapping
- Base random seed: 2026

## Parameter grid read from the source CSV files

- Initial cohesion: 3000, 3400, ..., 6600
- Friction angle: 18, 19, ..., 27
- log10 permeability: -2.3, -2.077778, -1.855556, -1.633333,
  -1.411111, -1.188889, -0.966667, -0.744444, -0.522222, -0.3

The manifest files retain higher-precision parameter values and map each
PNG to its source run and CSV file.

## Verification

All 5,000 PNG files were opened and verified after generation. No
corrupt images, missing runs, duplicate run identifiers, unexpected
dimensions, or non-RGB outputs were found.
