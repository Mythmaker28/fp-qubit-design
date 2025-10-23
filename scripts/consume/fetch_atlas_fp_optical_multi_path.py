"""
Try multiple paths to find atlas_fp_optical.csv (v1.2.1 / v1.3 / releases)
"""
import requests
import hashlib
from pathlib import Path
import json
import sys

REPO = "Mythmaker28/biological-qubits-atlas"
EXPECTED_SHA256 = "333adc871f5b2ec5118298de4e534a468c7379f053d8b03c13d7cd9eb7c43285"

# Paths to try
PATHS_TO_TRY = [
    # Releases
    ("release v1.3", f"https://api.github.com/repos/{REPO}/releases", "assets"),
    ("release v1.2.1", f"https://api.github.com/repos/{REPO}/releases/tags/v1.2.1", "direct"),
    # Branches
    ("branch: main", f"https://raw.githubusercontent.com/{REPO}/main/atlas_fp_optical.csv", "raw"),
    ("branch: v1.3", f"https://raw.githubusercontent.com/{REPO}/v1.3/atlas_fp_optical.csv", "raw"),
    ("branch: release/v1.3", f"https://raw.githubusercontent.com/{REPO}/release/v1.3/atlas_fp_optical.csv", "raw"),
    ("branch: v1.2.1", f"https://raw.githubusercontent.com/{REPO}/v1.2.1/atlas_fp_optical.csv", "raw"),
    ("branch: release/v1.2.1", f"https://raw.githubusercontent.com/{REPO}/release/v1.2.1/atlas_fp_optical.csv", "raw"),
    # Data folder
    ("main: data/", f"https://raw.githubusercontent.com/{REPO}/main/data/atlas_fp_optical.csv", "raw"),
    ("main: data/processed/", f"https://raw.githubusercontent.com/{REPO}/main/data/processed/atlas_fp_optical.csv", "raw"),
]

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "atlas_fp_optical.csv"
METADATA_PATH = OUTPUT_DIR / "TRAINING.METADATA.json"

def calculate_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()

def try_fetch():
    """Try all paths until one succeeds"""
    print("="*60)
    print("v1.1.4 MULTI-PATH FETCH")
    print("="*60)
    
    for attempt, (name, url, method) in enumerate(PATHS_TO_TRY, 1):
        print(f"\n[{attempt}/{len(PATHS_TO_TRY)}] Trying: {name}")
        print(f"     URL: {url}")
        
        try:
            if method == "raw":
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
                content = resp.content
                sha256 = calculate_sha256(content)
                
                print(f"     [OK] Downloaded {len(content)} bytes")
                print(f"     [SHA256] {sha256[:16]}...")
                
                return content, sha256, name, url
                
            elif method == "assets":
                # List all releases, find atlas_fp_optical.csv in assets
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
                releases = resp.json()
                
                for release in releases:
                    for asset in release.get('assets', []):
                        if 'atlas_fp_optical' in asset['name'].lower():
                            download_url = asset['browser_download_url']
                            print(f"     [OK] Found in {release['tag_name']}")
                            print(f"     [->] Downloading {asset['name']}...")
                            
                            asset_resp = requests.get(download_url, timeout=30)
                            asset_resp.raise_for_status()
                            content = asset_resp.content
                            sha256 = calculate_sha256(content)
                            
                            print(f"     [OK] Downloaded {len(content)} bytes")
                            print(f"     [SHA256] {sha256[:16]}...")
                            
                            return content, sha256, f"release {release['tag_name']}", download_url
                
                print(f"     [SKIP] No atlas_fp_optical.csv in releases")
                
            elif method == "direct":
                # Direct release tag
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
                release = resp.json()
                
                for asset in release.get('assets', []):
                    if 'atlas_fp_optical' in asset['name'].lower():
                        download_url = asset['browser_download_url']
                        print(f"     [OK] Found asset: {asset['name']}")
                        print(f"     [->] Downloading...")
                        
                        asset_resp = requests.get(download_url, timeout=30)
                        asset_resp.raise_for_status()
                        content = asset_resp.content
                        sha256 = calculate_sha256(content)
                        
                        print(f"     [OK] Downloaded {len(content)} bytes")
                        print(f"     [SHA256] {sha256[:16]}...")
                        
                        return content, sha256, name, download_url
                
                print(f"     [SKIP] No atlas_fp_optical.csv in this release")
        
        except requests.exceptions.RequestException as e:
            print(f"     [FAIL] {type(e).__name__}: {str(e)[:50]}")
            continue
    
    print("\n" + "="*60)
    print("[FAIL] File not found in any of the attempted paths")
    print("="*60)
    return None, None, None, None

def save_data(content: bytes, sha256: str, source_name: str, source_url: str):
    """Save CSV and metadata"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    OUTPUT_PATH.write_bytes(content)
    print(f"\n[OK] Saved to {OUTPUT_PATH}")
    
    metadata = {
        "source": "github_multi_path",
        "repo": REPO,
        "source_name": source_name,
        "source_url": source_url,
        "file": "atlas_fp_optical.csv",
        "sha256": sha256,
        "expected_sha256": EXPECTED_SHA256,
        "sha256_match": sha256.lower() == EXPECTED_SHA256.lower(),
        "size_bytes": len(content),
        "path": str(OUTPUT_PATH),
        "date": "2025-10-24"
    }
    
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f"[OK] Metadata saved to {METADATA_PATH}")
    
    return OUTPUT_PATH

def main():
    content, sha256, source_name, source_url = try_fetch()
    
    if content is None:
        print("\n[ACTION REQUIRED] Please provide the direct URL to atlas_fp_optical.csv")
        print("Or place the file manually in: data/processed/atlas_fp_optical.csv")
        sys.exit(1)
    
    csv_path = save_data(content, sha256, source_name, source_url)
    
    print("\n" + "="*60)
    print("[SUCCESS] File found and downloaded!")
    print(f"Source: {source_name}")
    print(f"File: {csv_path}")
    print(f"SHA256: {sha256}")
    if sha256.lower() == EXPECTED_SHA256.lower():
        print("[OK] SHA256 MATCHES! ✓")
    else:
        print("[WARN] SHA256 differs (file might be updated)")
    print("="*60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

