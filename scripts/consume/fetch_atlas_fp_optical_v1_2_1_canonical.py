"""
Fetch atlas_fp_optical.csv from biological-qubits-atlas v1.2.1 release
Canonical source (Chemin A)
"""
import requests
import hashlib
from pathlib import Path
import json
import sys

REPO = "Mythmaker28/biological-qubits-atlas"
TARGET_FILE = "atlas_fp_optical.csv"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
METADATA_PATH = OUTPUT_DIR / "TRAINING.METADATA.json"

def fetch_release_asset():
    """Try to fetch from v1.2.1 release"""
    print("[->] Fetching releases from GitHub API...")
    
    url = f"https://api.github.com/repos/{REPO}/releases"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        releases = resp.json()
    except Exception as e:
        print(f"[FAIL] GitHub API error: {e}")
        return None, None
    
    # Find v1.2.1
    for release in releases:
        if release['tag_name'] == 'v1.2.1':
            print(f"[OK] Found release v1.2.1")
            # Check assets
            for asset in release.get('assets', []):
                if asset['name'] == TARGET_FILE:
                    download_url = asset['browser_download_url']
                    print(f"[OK] Found asset: {TARGET_FILE}")
                    print(f"[->] Downloading from {download_url}...")
                    
                    try:
                        data_resp = requests.get(download_url, timeout=30)
                        data_resp.raise_for_status()
                        content = data_resp.content
                        
                        # Calculate SHA256
                        sha256 = hashlib.sha256(content).hexdigest()
                        print(f"[OK] Downloaded {len(content)} bytes")
                        print(f"[SHA256] {sha256}")
                        
                        return content, sha256
                    except Exception as e:
                        print(f"[FAIL] Download error: {e}")
                        return None, None
            
            print(f"[WARN] Asset {TARGET_FILE} not found in release v1.2.1")
            return None, None
    
    print("[WARN] Release v1.2.1 not found")
    return None, None

def save_data(content, sha256):
    """Save CSV and update metadata"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    csv_path = OUTPUT_DIR / TARGET_FILE
    csv_path.write_bytes(content)
    print(f"[OK] Saved to {csv_path}")
    
    # Update metadata
    metadata = {
        "source": "canonical",
        "repo": REPO,
        "release": "v1.2.1",
        "file": TARGET_FILE,
        "sha256": sha256,
        "path": str(csv_path),
        "method": "Chemin A (GitHub Release)"
    }
    
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f"[OK] Metadata saved to {METADATA_PATH}")
    
    return csv_path

def main():
    print("="*60)
    print("v1.1.4 RESUME - Chemin A (Canonique)")
    print("="*60)
    
    content, sha256 = fetch_release_asset()
    
    if content is None:
        print("\n[FAIL] Chemin A failed. Use Chemin B (fallback local).")
        sys.exit(1)
    
    csv_path = save_data(content, sha256)
    
    print("\n" + "="*60)
    print("[SUCCESS] Chemin A completed!")
    print(f"File: {csv_path}")
    print(f"SHA256: {sha256}")
    print("="*60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())


