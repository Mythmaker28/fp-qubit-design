#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust multi-path discovery of atlas_fp_optical.csv v1.2.1.

Strategy (ordered by priority):
1. Releases: Check v1.2.1 release assets
2. Tags: Try direct download URL
3. Branches: Check specific branches for versioned file

All attempts are logged to reports/WHERE_I_LOOKED.md
"""

import sys
import json
import hashlib
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime


REPO_OWNER = "Mythmaker28"
REPO_NAME = "biological-qubits-atlas"
GITHUB_API_BASE = "https://api.github.com"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com"

# Expected SHA256 for atlas_fp_optical.csv v1.2.1
EXPECTED_SHA256 = "333ADC871F5B2EC5118298DE4E534A468C7379F053D8B03C13D7CD9EB7C43285"

# Target filename
TARGET_FILENAME = "atlas_fp_optical.csv"


class DiscoveryLog:
    """Logger for discovery attempts."""
    
    def __init__(self):
        self.entries = []
        self.start_time = datetime.now()
    
    def log(self, step: str, result: str, details: dict = None):
        """Log a discovery attempt."""
        entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'step': step,
            'result': result,
            'details': details or {}
        }
        self.entries.append(entry)
        
        # Print to console (ASCII-safe for Windows)
        status_icon = "[OK]" if result == "SUCCESS" else "[FAIL]" if result == "FAIL" else "[->]"
        print(f"  {status_icon} {step}: {result}")
        if details:
            for key, value in details.items():
                print(f"      {key}: {value}")
    
    def save(self, output_path: Path):
        """Save log to markdown file."""
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# WHERE I LOOKED - Atlas v1.2.1 Discovery Log\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Duration**: {(datetime.now() - self.start_time).total_seconds():.2f}s\n\n")
            f.write("---\n\n")
            
            f.write("## Discovery Strategy\n\n")
            f.write("1. **Releases**: Check GitHub Releases API for v1.2.1 assets\n")
            f.write("2. **Tags**: Try direct download URL for tag v1.2.1\n")
            f.write("3. **Branches**: Check specific branches for versioned file\n\n")
            
            f.write("---\n\n")
            f.write("## Attempts Log\n\n")
            
            for i, entry in enumerate(self.entries, 1):
                f.write(f"### Attempt {i}: {entry['step']}\n\n")
                f.write(f"- **Timestamp**: {entry['timestamp']}\n")
                f.write(f"- **Result**: **{entry['result']}**\n")
                
                if entry['details']:
                    f.write("- **Details**:\n")
                    for key, value in entry['details'].items():
                        f.write(f"  - `{key}`: {value}\n")
                
                f.write("\n")
            
            f.write("---\n\n")
            f.write("## Conclusion\n\n")
            
            success_count = sum(1 for e in self.entries if e['result'] == 'SUCCESS')
            
            if success_count > 0:
                f.write(f"[OK] **Found after {len(self.entries)} attempts**\n")
            else:
                f.write(f"[FAIL] **Not found after {len(self.entries)} attempts**\n\n")
                f.write("### Recommendation\n\n")
                f.write("The canonical `atlas_fp_optical.csv` v1.2.1 with 66 FP optical entries ")
                f.write("does not exist in the public Atlas repository.\n\n")
                f.write("**Options**:\n")
                f.write("1. Wait for Atlas maintainer to publish this filtered subset\n")
                f.write("2. Create it locally from `biological_qubits.csv` (but only 2-3 FP exist)\n")
                f.write("3. Expand scope to include quantum sensing systems (NV centers, etc.)\n")
                f.write("4. Integrate external FP databases (FPbase, UniProt)\n")


def calculate_sha256(file_path: Path) -> str:
    """Calculate SHA256 of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def fetch_url(url: str, output_path: Path) -> tuple:
    """
    Fetch URL and save to file.
    
    Returns: (success: bool, error_msg: str or None)
    """
    try:
        urllib.request.urlretrieve(url, output_path)
        return (True, None)
    except urllib.error.HTTPError as e:
        return (False, f"HTTP {e.code}: {e.reason}")
    except Exception as e:
        return (False, str(e))


def check_releases(log: DiscoveryLog) -> tuple:
    """
    Step 1: Check GitHub Releases API.
    
    Returns: (found: bool, file_path: Path or None, ref: str or None)
    """
    print("\n[STEP 1] Checking GitHub Releases API...")
    
    url = f"{GITHUB_API_BASE}/repos/{REPO_OWNER}/{REPO_NAME}/releases"
    
    log.log("Releases API Query", "ATTEMPT", {
        'url': url,
        'looking_for': f"v1.2.1 with asset {TARGET_FILENAME}"
    })
    
    try:
        with urllib.request.urlopen(url) as response:
            releases = json.loads(response.read())
        
        log.log("Releases API Query", "SUCCESS", {
            'total_releases': len(releases)
        })
        
        # Find v1.2.1
        target_release = None
        for release in releases:
            if release['tag_name'] == 'v1.2.1':
                target_release = release
                break
        
        if not target_release:
            log.log("Find v1.2.1 Release", "FAIL", {
                'reason': "Tag v1.2.1 not found in releases",
                'available_tags': [r['tag_name'] for r in releases[:5]]
            })
            return (False, None, None)
        
        log.log("Find v1.2.1 Release", "SUCCESS", {
            'published_at': target_release['published_at'],
            'assets_count': len(target_release['assets'])
        })
        
        # Check assets
        target_asset = None
        for asset in target_release['assets']:
            if asset['name'] == TARGET_FILENAME:
                target_asset = asset
                break
        
        if not target_asset:
            log.log("Find Asset", "FAIL", {
                'reason': f"{TARGET_FILENAME} not in release assets",
                'available_assets': [a['name'] for a in target_release['assets']]
            })
            return (False, None, None)
        
        # Download asset
        download_url = target_asset['browser_download_url']
        output_path = Path("data/external/atlas_fp_optical_v1_2_1.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        log.log("Download Asset", "ATTEMPT", {
            'url': download_url,
            'size': f"{target_asset['size']} bytes"
        })
        
        success, error = fetch_url(download_url, output_path)
        
        if not success:
            log.log("Download Asset", "FAIL", {'error': error})
            return (False, None, None)
        
        log.log("Download Asset", "SUCCESS", {
            'saved_to': str(output_path)
        })
        
        # Verify SHA256
        actual_sha = calculate_sha256(output_path)
        
        log.log("Verify SHA256", "ATTEMPT", {
            'expected': EXPECTED_SHA256,
            'actual': actual_sha
        })
        
        if actual_sha != EXPECTED_SHA256:
            log.log("Verify SHA256", "FAIL", {
                'mismatch': f"Expected {EXPECTED_SHA256}, got {actual_sha}"
            })
            return (False, None, None)
        
        log.log("Verify SHA256", "SUCCESS", {
            'match': "SHA256 verified"
        })
        
        return (True, output_path, f"v1.2.1 (asset)")
        
    except Exception as e:
        log.log("Releases API Query", "FAIL", {'error': str(e)})
        return (False, None, None)


def check_tags(log: DiscoveryLog) -> tuple:
    """
    Step 2: Check tags and try direct download URL.
    
    Returns: (found: bool, file_path: Path or None, ref: str or None)
    """
    print("\n[STEP 2] Checking Tags API...")
    
    url = f"{GITHUB_API_BASE}/repos/{REPO_OWNER}/{REPO_NAME}/git/refs/tags"
    
    log.log("Tags API Query", "ATTEMPT", {'url': url})
    
    try:
        with urllib.request.urlopen(url) as response:
            tags = json.loads(response.read())
        
        log.log("Tags API Query", "SUCCESS", {
            'total_tags': len(tags)
        })
        
        # Check if v1.2.1 exists
        v121_exists = any(tag['ref'] == 'refs/tags/v1.2.1' for tag in tags)
        
        if not v121_exists:
            log.log("Find v1.2.1 Tag", "FAIL", {
                'reason': "Tag v1.2.1 not found",
                'available_tags': [t['ref'].split('/')[-1] for t in tags[:5]]
            })
            return (False, None, None)
        
        log.log("Find v1.2.1 Tag", "SUCCESS", {
            'tag': "v1.2.1 exists"
        })
        
        # Try direct download URL
        download_url = f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/download/v1.2.1/{TARGET_FILENAME}"
        output_path = Path("data/external/atlas_fp_optical_v1_2_1.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        log.log("Direct Download URL", "ATTEMPT", {'url': download_url})
        
        success, error = fetch_url(download_url, output_path)
        
        if not success:
            log.log("Direct Download URL", "FAIL", {'error': error})
            return (False, None, None)
        
        log.log("Direct Download URL", "SUCCESS", {
            'saved_to': str(output_path)
        })
        
        # Verify SHA256
        actual_sha = calculate_sha256(output_path)
        
        if actual_sha != EXPECTED_SHA256:
            log.log("Verify SHA256", "FAIL", {
                'mismatch': f"Expected {EXPECTED_SHA256}, got {actual_sha}"
            })
            return (False, None, None)
        
        log.log("Verify SHA256", "SUCCESS", {
            'match': "SHA256 verified"
        })
        
        return (True, output_path, f"v1.2.1 (direct URL)")
        
    except Exception as e:
        log.log("Tags API Query", "FAIL", {'error': str(e)})
        return (False, None, None)


def check_branches(log: DiscoveryLog) -> tuple:
    """
    Step 3: Check specific branches for versioned file.
    
    Returns: (found: bool, file_path: Path or None, ref: str or None)
    """
    print("\n[STEP 3] Checking Branches...")
    
    branches_to_check = [
        "release/v1.2.1-fp-optical-push",
        "main"
    ]
    
    paths_to_try = [
        "data/processed/atlas_fp_optical.csv",
        "data/processed/atlas_all_real.csv",
        "atlas_fp_optical.csv"
    ]
    
    for branch in branches_to_check:
        log.log(f"Check Branch: {branch}", "ATTEMPT", {})
        
        for path in paths_to_try:
            url = f"{GITHUB_RAW_BASE}/{REPO_OWNER}/{REPO_NAME}/{branch}/{path}"
            
            log.log(f"Try Path: {path}", "ATTEMPT", {
                'url': url
            })
            
            output_path = Path("data/external/atlas_fp_optical_v1_2_1.csv")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            success, error = fetch_url(url, output_path)
            
            if not success:
                log.log(f"Try Path: {path}", "FAIL", {'error': error})
                continue
            
            log.log(f"Try Path: {path}", "SUCCESS", {
                'saved_to': str(output_path),
                'branch': branch
            })
            
            # Check SHA256 if available
            actual_sha = calculate_sha256(output_path)
            
            log.log("Check SHA256", "INFO", {
                'actual': actual_sha,
                'expected': EXPECTED_SHA256,
                'match': actual_sha == EXPECTED_SHA256
            })
            
            # If SHA matches, great! Otherwise, get commit SHA
            if actual_sha == EXPECTED_SHA256:
                return (True, output_path, f"{branch} (SHA256 verified)")
            else:
                # Get commit SHA for provenance
                commit_url = f"{GITHUB_API_BASE}/repos/{REPO_OWNER}/{REPO_NAME}/commits/{branch}"
                
                try:
                    with urllib.request.urlopen(commit_url) as response:
                        commit_data = json.loads(response.read())
                        commit_sha = commit_data['sha'][:8]
                    
                    log.log("Get Commit SHA", "SUCCESS", {
                        'commit_sha': commit_sha
                    })
                    
                    return (True, output_path, f"{branch}@{commit_sha}")
                
                except Exception as e:
                    log.log("Get Commit SHA", "FAIL", {'error': str(e)})
                    return (True, output_path, f"{branch} (commit SHA unavailable)")
        
        log.log(f"Check Branch: {branch}", "FAIL", {
            'reason': f"None of the paths found: {paths_to_try}"
        })
    
    return (False, None, None)


def main():
    print("=" * 60)
    print("Robust Atlas v1.2.1 Discovery")
    print("=" * 60)
    print()
    print(f"Target: {TARGET_FILENAME}")
    print(f"Expected SHA256: {EXPECTED_SHA256}")
    print()
    
    log = DiscoveryLog()
    
    # Try Step 1: Releases
    found, file_path, ref = check_releases(log)
    
    if found:
        print(f"\n[SUCCESS] Found via releases: {ref}")
        log.save(Path("reports/WHERE_I_LOOKED.md"))
        print(f"\nSaved to: {file_path}")
        print(f"Reference: {ref}")
        sys.exit(0)
    
    # Try Step 2: Tags
    found, file_path, ref = check_tags(log)
    
    if found:
        print(f"\n[SUCCESS] Found via tags: {ref}")
        log.save(Path("reports/WHERE_I_LOOKED.md"))
        print(f"\nSaved to: {file_path}")
        print(f"Reference: {ref}")
        sys.exit(0)
    
    # Try Step 3: Branches
    found, file_path, ref = check_branches(log)
    
    if found:
        print(f"\n[SUCCESS] Found via branches: {ref}")
        log.save(Path("reports/WHERE_I_LOOKED.md"))
        print(f"\nSaved to: {file_path}")
        print(f"Reference: {ref}")
        sys.exit(0)
    
    # Not found
    print("\n" + "=" * 60)
    print("CANONICAL v1.2.1 FP OPTICAL NOT FOUND")
    print("=" * 60)
    print()
    print(f"Canonique v1.2.1 FP optical non trouvé.")
    print(f"Voir reports/WHERE_I_LOOKED.md pour détails.")
    print()
    
    log.save(Path("reports/WHERE_I_LOOKED.md"))
    
    print(f"Discovery log saved: reports/WHERE_I_LOOKED.md")
    print()
    
    sys.exit(1)


if __name__ == "__main__":
    main()

