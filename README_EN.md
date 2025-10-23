# FP-Qubit Design

## Purpose

Software framework for **in silico design of fluorescent protein (FP) mutants** optimized for biological qubit-related photophysical proxies (coherence, contrast). No wet-lab experiments, purely computational.

**Status**: Skeleton (v0.1.0) — structure and TODOs only, no trained ML models yet.

## Context

- **Parent project**: [Biological Qubits Atlas](https://github.com/Mythmaker28/biological-qubits-atlas) — dataset of ~22 quantum systems in biological contexts (T1/T2, contrast, provenance; CC BY 4.0 license).
- **Approach**: Use Atlas photophysical proxies (lifetime, contrast, temperature) to guide FP mutant design.
- **Target**: GFP-like fluorescent proteins with enhanced quantum coherence and photostability properties.
- **Publication**: Zenodo + GitHub Pages (shortlist table).

## Data provenance

Proxies based on Atlas snapshot:
- **Source**: https://github.com/Mythmaker28/biological-qubits-atlas
- **Commit**: `abd6a4cd7dde94dc4ca7cde69aee3fad25757bcf`
- **Schema**: v1.2 (~33 columns)
- **License**: CC BY 4.0
- **Local snapshot**: `data/processed/atlas_snapshot.csv` (read-only)

See `data/processed/atlas_snapshot.METADATA.json` for full metadata.

## Install

```bash
git clone https://github.com/Mythmaker28/fp-qubit-design.git
cd fp-qubit-design
pip install -r requirements.txt
```

**Dependencies**: numpy, pandas, scikit-learn, matplotlib (Python ≥3.8).

## Quickstart (skeleton)

```bash
# Generate mutants (TODO script)
python scripts/generate_mutants.py --config configs/example.yaml

# Train baseline (TODO script)
python scripts/train_baseline.py --config configs/example.yaml
```

**Note**: Scripts are placeholders with TODOs (no actual training yet).

## Roadmap

### 30 days
- Define Atlas → FP proxy mapping
- Implement basic featurization (AA composition, physicochemical properties)
- RF/XGB baseline proof-of-concept
- Generate first mutant candidates

### 60 days
- Cross-validation of baselines
- Uncertainty quantification (bootstrap/GP)
- Shortlist 10-20 "qubit-friendly" mutants
- Publish web page (GitHub Pages)

### 90 days
- (Optional) GNN prototype
- Zenodo publication with DOI
- Complete documentation (IMRaD)
- Open to external contributions

## License & Citation

- **Code**: Apache-2.0 (see `LICENSE`)
- **Atlas data**: CC BY 4.0 (see Atlas repo)

If you use this repo, please cite:

```
Lepesteur, T. (2025). FP-Qubit Design (v0.1.0). GitHub. https://github.com/Mythmaker28/fp-qubit-design
```

See `CITATION.cff` for structured format.

## Contact

- **Author**: Tommy Lepesteur
- **ORCID**: [0009-0009-0577-9563](https://orcid.org/0009-0009-0577-9563)
- **Issues**: https://github.com/Mythmaker28/fp-qubit-design/issues

---

**Status**: 🚧 Skeleton (v0.1.0) — Under active development

