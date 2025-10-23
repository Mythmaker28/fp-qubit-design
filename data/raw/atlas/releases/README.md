# Atlas Releases - Raw Data

## Provenance

This directory contains raw CSV/TSV/JSON assets downloaded from **ALL releases** of the [Biological Qubits Atlas](https://github.com/Mythmaker28/biological-qubits-atlas).

## Structure

```
releases/
├─ v1.0/
│  ├─ biological_qubits.csv
│  └─ ...
├─ v1.1/
│  └─ ...
└─ v1.2/
   └─ ...
```

## Harvest Process

Assets are downloaded via `scripts/etl/fetch_atlas_releases.py` using the GitHub API.

For each release:
- **Tag**: Git tag (e.g., `v1.2.0`)
- **Published**: Release date
- **Assets**: All CSV/TSV/JSON files attached
- **SHA256**: Checksum for integrity verification

## License

Data sourced from Biological Qubits Atlas is licensed under **CC BY 4.0**.

**Citation**:
Lepesteur, T. (2025). Biological Qubits Atlas. GitHub. https://github.com/Mythmaker28/biological-qubits-atlas

## Processing

Raw assets are merged and normalized by `scripts/etl/merge_atlas_assets.py` into `data/interim/atlas_merged.parquet`.

---

**DO NOT MODIFY** files in this directory. They are pristine copies from upstream releases.

