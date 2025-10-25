# fp-qubit-design

**Status: Documentation-only; modeling release is NO-GO**

Design of fluorescent protein (FP) mutants as candidate biological qubits. Focus: coherence-compatible reporters with usable optical contrast. This repository currently ships **lab handoff materials** only. Modeling targets (R^2 ≥ 0.20 and MAE < 7.81) were **not met**; uncertainty calibration acceptable via CQR-like methods; result: **no model release**.

## Current Status (Atlas v2.2.2)
- **Atlas version**: v2.2.2
- **Useful systems**: 221
- **Families**: 30
- **Ca2+ share**: 22.6%
- **Decision**: Proceed with lab shortlists; **no** public model.
- **Pages**: see GitHub Pages for live counts.

If `deliverables/lab_v2_2_2/status.json` is present, authoritative counts are read there. If absent, the above defaults apply.

## Deliverables (lab_v2_2_2)
Located under `deliverables/lab_v2_2_2/` (if present):
- `shortlist_lab_sheet.csv`
- `shortlist_top12_final.csv`
- `plate_layout_96.csv`
- `plate_layout_24.csv`
- `protocol_skeleton.md`
- `filters_recommendations.md`
- `SHA256SUMS.txt`
- `status.json`
- `lab_v2_2_2.zip`

## Lab Usage (handoff)
1. Use `shortlist_lab_sheet.csv` for procurement and tracking.
2. Plate according to `plate_layout_96.csv` or `plate_layout_24.csv`.
3. Follow `protocol_skeleton.md`; consult `filters_recommendations.md` for optics.
4. Verify integrity using `SHA256SUMS.txt`.

## Evaluation Note
Strict acceptance criteria for modeling were not achieved (R^2 and MAE thresholds). Coverage and calibration were acceptable with conformal approaches, but **release is deferred** until targets are met.

## Repository Structure
- `deliverables/` – lab packages and status.
- `reports/` – data and release notes.
- `scripts/` – assembly utilities (no training code changes here).
- `index.html` – Pages landing (static; reads `status.json` if present).

## Roadmap → v2.3
- Expand features: quantum efficiency (QE), extinction coefficient (EC), brightness; photostability.
- More modalities (Voltage, neurotransmitters).
- Model routers/stacking; calibrated CQR.

## License & Citation
- License: MIT (unless otherwise noted).
- Cite as: "fp-qubit-design (v2.2.2), lab handoff package" and link to this repository.

## Contact
Open an issue on GitHub.
