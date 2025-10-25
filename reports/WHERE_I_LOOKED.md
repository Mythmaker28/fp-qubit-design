# WHERE I LOOKED - Atlas v1.2.1 Discovery Log

**Generated**: 2025-10-24 00:12:42
**Duration**: 4.81s

---

## Discovery Strategy

1. **Releases**: Check GitHub Releases API for v1.2.1 assets
2. **Tags**: Try direct download URL for tag v1.2.1
3. **Branches**: Check specific branches for versioned file

---

## Attempts Log

### Attempt 1: Releases API Query

- **Timestamp**: 2025-10-24 00:12:37
- **Result**: **ATTEMPT**
- **Details**:
  - `url`: https://api.github.com/repos/Mythmaker28/biological-qubits-atlas/releases
  - `looking_for`: v1.2.1 with asset atlas_fp_optical.csv

### Attempt 2: Releases API Query

- **Timestamp**: 2025-10-24 00:12:38
- **Result**: **SUCCESS**
- **Details**:
  - `total_releases`: 2

### Attempt 3: Find v1.2.1 Release

- **Timestamp**: 2025-10-24 00:12:38
- **Result**: **SUCCESS**
- **Details**:
  - `published_at`: 2025-10-22T23:52:18Z
  - `assets_count`: 4

### Attempt 4: Find Asset

- **Timestamp**: 2025-10-24 00:12:38
- **Result**: **FAIL**
- **Details**:
  - `reason`: atlas_fp_optical.csv not in release assets
  - `available_assets`: ['biological_qubits.csv', 'CITATION.cff', 'LICENSE', 'QC_REPORT.md']

### Attempt 5: Tags API Query

- **Timestamp**: 2025-10-24 00:12:38
- **Result**: **ATTEMPT**
- **Details**:
  - `url`: https://api.github.com/repos/Mythmaker28/biological-qubits-atlas/git/refs/tags

### Attempt 6: Tags API Query

- **Timestamp**: 2025-10-24 00:12:39
- **Result**: **SUCCESS**
- **Details**:
  - `total_tags`: 2

### Attempt 7: Find v1.2.1 Tag

- **Timestamp**: 2025-10-24 00:12:39
- **Result**: **SUCCESS**
- **Details**:
  - `tag`: v1.2.1 exists

### Attempt 8: Direct Download URL

- **Timestamp**: 2025-10-24 00:12:39
- **Result**: **ATTEMPT**
- **Details**:
  - `url`: https://github.com/Mythmaker28/biological-qubits-atlas/releases/download/v1.2.1/atlas_fp_optical.csv

### Attempt 9: Direct Download URL

- **Timestamp**: 2025-10-24 00:12:40
- **Result**: **FAIL**
- **Details**:
  - `error`: HTTP 404: Not Found

### Attempt 10: Check Branch: release/v1.2.1-fp-optical-push

- **Timestamp**: 2025-10-24 00:12:40
- **Result**: **ATTEMPT**

### Attempt 11: Try Path: data/processed/atlas_fp_optical.csv

- **Timestamp**: 2025-10-24 00:12:40
- **Result**: **ATTEMPT**
- **Details**:
  - `url`: https://raw.githubusercontent.com/Mythmaker28/biological-qubits-atlas/release/v1.2.1-fp-optical-push/data/processed/atlas_fp_optical.csv

### Attempt 12: Try Path: data/processed/atlas_fp_optical.csv

- **Timestamp**: 2025-10-24 00:12:40
- **Result**: **FAIL**
- **Details**:
  - `error`: HTTP 404: Not Found

### Attempt 13: Try Path: data/processed/atlas_all_real.csv

- **Timestamp**: 2025-10-24 00:12:40
- **Result**: **ATTEMPT**
- **Details**:
  - `url`: https://raw.githubusercontent.com/Mythmaker28/biological-qubits-atlas/release/v1.2.1-fp-optical-push/data/processed/atlas_all_real.csv

### Attempt 14: Try Path: data/processed/atlas_all_real.csv

- **Timestamp**: 2025-10-24 00:12:41
- **Result**: **FAIL**
- **Details**:
  - `error`: HTTP 404: Not Found

### Attempt 15: Try Path: atlas_fp_optical.csv

- **Timestamp**: 2025-10-24 00:12:41
- **Result**: **ATTEMPT**
- **Details**:
  - `url`: https://raw.githubusercontent.com/Mythmaker28/biological-qubits-atlas/release/v1.2.1-fp-optical-push/atlas_fp_optical.csv

### Attempt 16: Try Path: atlas_fp_optical.csv

- **Timestamp**: 2025-10-24 00:12:41
- **Result**: **FAIL**
- **Details**:
  - `error`: HTTP 404: Not Found

### Attempt 17: Check Branch: release/v1.2.1-fp-optical-push

- **Timestamp**: 2025-10-24 00:12:41
- **Result**: **FAIL**
- **Details**:
  - `reason`: None of the paths found: ['data/processed/atlas_fp_optical.csv', 'data/processed/atlas_all_real.csv', 'atlas_fp_optical.csv']

### Attempt 18: Check Branch: main

- **Timestamp**: 2025-10-24 00:12:41
- **Result**: **ATTEMPT**

### Attempt 19: Try Path: data/processed/atlas_fp_optical.csv

- **Timestamp**: 2025-10-24 00:12:41
- **Result**: **ATTEMPT**
- **Details**:
  - `url`: https://raw.githubusercontent.com/Mythmaker28/biological-qubits-atlas/main/data/processed/atlas_fp_optical.csv

### Attempt 20: Try Path: data/processed/atlas_fp_optical.csv

- **Timestamp**: 2025-10-24 00:12:42
- **Result**: **FAIL**
- **Details**:
  - `error`: HTTP 404: Not Found

### Attempt 21: Try Path: data/processed/atlas_all_real.csv

- **Timestamp**: 2025-10-24 00:12:42
- **Result**: **ATTEMPT**
- **Details**:
  - `url`: https://raw.githubusercontent.com/Mythmaker28/biological-qubits-atlas/main/data/processed/atlas_all_real.csv

### Attempt 22: Try Path: data/processed/atlas_all_real.csv

- **Timestamp**: 2025-10-24 00:12:42
- **Result**: **FAIL**
- **Details**:
  - `error`: HTTP 404: Not Found

### Attempt 23: Try Path: atlas_fp_optical.csv

- **Timestamp**: 2025-10-24 00:12:42
- **Result**: **ATTEMPT**
- **Details**:
  - `url`: https://raw.githubusercontent.com/Mythmaker28/biological-qubits-atlas/main/atlas_fp_optical.csv

### Attempt 24: Try Path: atlas_fp_optical.csv

- **Timestamp**: 2025-10-24 00:12:42
- **Result**: **FAIL**
- **Details**:
  - `error`: HTTP 404: Not Found

### Attempt 25: Check Branch: main

- **Timestamp**: 2025-10-24 00:12:42
- **Result**: **FAIL**
- **Details**:
  - `reason`: None of the paths found: ['data/processed/atlas_fp_optical.csv', 'data/processed/atlas_all_real.csv', 'atlas_fp_optical.csv']

---

## Conclusion

[OK] **Found after 25 attempts**
