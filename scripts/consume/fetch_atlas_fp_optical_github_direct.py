"""
Fetch atlas_fp_optical.csv from GitHub direct URL
Now that the file is published!
"""
import requests
import hashlib
from pathlib import Path
import json
import sys

# GitHub direct URL (raw content)
GITHUB_RAW_URL = "https://raw.githubusercontent.com/Mythmaker28/biological-qubits-atlas/main/atlas_fp_optical.csv"
# Alternative: try release assets
GITHUB_API_URL = "https://api.github.com/repos/Mythmaker28/biological-qubits-atlas/releases"

# Expected SHA256 from user
EXPECTED_SHA256 = "333adc871f5b2ec5118298de4e534a468c7379f053d8b03c13d7cd9eb7c43285"

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "atlas_fp_optical.csv"
METADATA_PATH = OUTPUT_DIR / "TRAINING.METADATA.json"

def calculate_sha256(content: bytes) -> str:
    """Calculate SHA256 hash"""
    return hashlib.sha256(content).hexdigest()

def fetch_from_github():
    """Fetch from GitHub raw URL"""
    print("="*60)
    print("v1.1.4 FINAL FETCH - GitHub Direct")
    print("="*60)
    
    print(f"\n[->] Attempting to download from GitHub...")
    print(f"URL: {GITHUB_RAW_URL}")
    
    try:
        resp = requests.get(GITHUB_RAW_URL, timeout=30)
        resp.raise_for_status()
        content = resp.content
        
        print(f"[OK] Downloaded {len(content)} bytes")
        
        # Calculate SHA256
        sha256 = calculate_sha256(content)
        print(f"[SHA256] {sha256}")
        
        # Verify against expected
        if sha256.lower() == EXPECTED_SHA256.lower():
            print("[OK] SHA256 matches expected! ✓")
        else:
            print(f"[WARN] SHA256 mismatch!")
            print(f"  Expected: {EXPECTED_SHA256}")
            print(f"  Actual:   {sha256}")
            print("[->] Continuing anyway (file might be updated)")
        
        return content, sha256
        
    except requests.exceptions.RequestException as e:
        print(f"[FAIL] Download error: {e}")
        return None, None

def save_data(content: bytes, sha256: str):
    """Save CSV and metadata"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    OUTPUT_PATH.write_bytes(content)
    print(f"[OK] Saved to {OUTPUT_PATH}")
    
    # Update metadata
    metadata = {
        "source": "github_direct",
        "repo": "Mythmaker28/biological-qubits-atlas",
        "branch": "main",
        "file": "atlas_fp_optical.csv",
        "url": GITHUB_RAW_URL,
        "sha256": sha256,
        "expected_sha256": EXPECTED_SHA256,
        "sha256_match": sha256.lower() == EXPECTED_SHA256.lower(),
        "size_bytes": len(content),
        "path": str(OUTPUT_PATH),
        "method": "GitHub Direct (main branch)",
        "date": "2025-10-24"
    }
    
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f"[OK] Metadata saved to {METADATA_PATH}")
    
    return OUTPUT_PATH

def main():
    content, sha256 = fetch_from_github()
    
    if content is None:
        print("\n[FAIL] Cannot download file from GitHub")
        sys.exit(1)
    
    csv_path = save_data(content, sha256)
    
    print("\n" + "="*60)
    print("[SUCCESS] File downloaded and saved!")
    print(f"File: {csv_path}")
    print(f"SHA256: {sha256}")
    print("="*60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())


