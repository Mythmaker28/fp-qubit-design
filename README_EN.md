# FP-Qubit Design

## Purpose

Software framework for **in silico design of fluorescent protein (FP) mutants** optimized for biological qubit-related photophysical proxies (coherence, contrast). No wet-lab experiments, purely computational.

**Status**: v1.0.0 Public Release — functional baseline ML, 30 optimized mutants, figures, and interactive website.

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

## Quickstart

```bash
# 1. Train baseline Random Forest model
python scripts/train_baseline.py --config configs/example.yaml

# 2. Generate mutant shortlist (30 candidates)
python scripts/generate_mutants.py --config configs/example.yaml

# 3. Generate figures
python scripts/generate_figures.py

# 4. View shortlist online
# https://mythmaker28.github.io/fp-qubit-design/ (once Pages enabled)
```

## Results (v1.0.0)

**Baseline ML**: Random Forest, Test MAE ~4.6%, CV MAE 4.79 ± 0.42%  
**Mutant Shortlist**: 30 candidates, predicted gain +2.1% to +12.3% (mean +4.0%)  
**Visualizations**: Feature importance, predicted gains histogram  

## Future Roadmap (v1.1+)

- Parse "Photophysique" field (lifetime, QY)
- Real ΔΔG calculations (FoldX or ML)
- 3D structures (PDB alignment)
- GNN prototype (optional)
- Zenodo DOI publication

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

**Status**: ✅ v1.0.0 Public Release — Fully functional

