"""
Fetch atlas_fp_optical.csv - Chemin B (Fallback Local)
Uses locally provided CSV file
"""
import hashlib
from pathlib import Path
import json
import sys
import shutil

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
FALLBACK_PATH = PROJECT_ROOT / "data" / "external" / "atlas_fp_optical_v1_2_1.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "atlas_fp_optical.csv"
METADATA_PATH = OUTPUT_DIR / "TRAINING.METADATA.json"
PROVENANCE_PATH = PROJECT_ROOT / "data" / "external" / "atlas" / "PROVENANCE.md"

def calculate_sha256(file_path: Path) -> str:
    """Calculate SHA256 hash of file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def main():
    print("="*60)
    print("v1.1.4 RESUME - Chemin B (Fallback Local)")
    print("="*60)
    
    # Check if fallback file exists
    if not FALLBACK_PATH.exists():
        print(f"\n[FAIL] Fallback file not found: {FALLBACK_PATH}")
        print("\nExpected path structure:")
        print("  data/external/atlas/atlas_fp_optical_v1_2_1.csv")
        print("\nPlease provide the file or create it manually.")
        sys.exit(1)
    
    print(f"[OK] Found fallback file: {FALLBACK_PATH}")
    
    # Calculate SHA256
    print("[->] Calculating SHA256...")
    sha256 = calculate_sha256(FALLBACK_PATH)
    print(f"[SHA256] {sha256}")
    
    # Copy to processed
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(FALLBACK_PATH, OUTPUT_PATH)
    print(f"[OK] Copied to {OUTPUT_PATH}")
    
    # Read size
    size_bytes = FALLBACK_PATH.stat().st_size
    print(f"[INFO] File size: {size_bytes} bytes")
    
    # Update metadata
    metadata = {
        "source": "fallback_local",
        "original_path": str(FALLBACK_PATH),
        "release": "v1.2.1",
        "file": "atlas_fp_optical.csv",
        "sha256": sha256,
        "size_bytes": size_bytes,
        "path": str(OUTPUT_PATH),
        "method": "Chemin B (Fallback Local)"
    }
    
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f"[OK] Metadata saved to {METADATA_PATH}")
    
    # Create provenance doc
    PROVENANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    provenance_content = f"""# Provenance: atlas_fp_optical.csv v1.2.1

**Source**: Fallback Local (Chemin B)

**Original Path**: `{FALLBACK_PATH}`

**SHA256**: `{sha256}`

**Size**: {size_bytes} bytes

**Method**: Chemin B (Fallback Local) - utilisé car l'asset n'était pas disponible dans la release GitHub v1.2.1.

**License**: CC BY 4.0 (assumed from biological-qubits-atlas)

**Date**: 2025-10-24
"""
    
    PROVENANCE_PATH.write_text(provenance_content, encoding='utf-8')
    print(f"[OK] Provenance saved to {PROVENANCE_PATH}")
    
    print("\n" + "="*60)
    print("[SUCCESS] Chemin B completed!")
    print(f"File: {OUTPUT_PATH}")
    print(f"SHA256: {sha256}")
    print("="*60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

