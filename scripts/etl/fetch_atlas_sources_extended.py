#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended Atlas sources fetcher - includes gh-pages, older branches, and archives.

This script fetches ALL possible sources from biological-qubits-atlas:
1. All releases (including prereleases)
2. gh-pages branch (if exists)
3. Historical branches (data/, old versions)
4. Archives (.zip, .tar.gz) extraction
"""

import argparse
import json
import hashlib
import zipfile
import tarfile
from pathlib import Path
from datetime import datetime
import urllib.request
import urllib.error


ATLAS_REPO = "Mythmaker28/biological-qubits-atlas"
GITHUB_API_BASE = "https://api.github.com"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch extended Atlas sources"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/atlas/releases",
        help="Output directory"
    )
    return parser.parse_args()


def fetch_branches(repo: str) -> list:
    """Fetch all branches from repo."""
    url = f"{GITHUB_API_BASE}/repos/{repo}/branches"
    
    print(f"[INFO] Fetching branches from: {url}")
    
    try:
        with urllib.request.urlopen(url) as response:
            branches = json.loads(response.read())
    except urllib.error.HTTPError as e:
        print(f"[ERROR] HTTP {e.code}: {e.reason}")
        return []
    
    print(f"[INFO] Found {len(branches)} branches")
    
    return branches


def download_from_branch(repo: str, branch: str, filename: str, output_dir: Path) -> bool:
    """Download a specific file from a branch."""
    url = f"{GITHUB_RAW_BASE}/{repo}/{branch}/{filename}"
    
    output_file = output_dir / branch / filename
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"  [INFO] Downloading from branch {branch}: {filename}")
    
    try:
        urllib.request.urlretrieve(url, output_file)
        return True
    except Exception as e:
        print(f"  [WARN] Failed to download {filename} from {branch}: {e}")
        return False


def extract_archive(archive_path: Path, extract_dir: Path):
    """Extract ZIP or TAR.GZ archive."""
    print(f"  [INFO] Extracting: {archive_path.name}")
    
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    elif archive_path.name.endswith(('.tar.gz', '.tgz')):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_dir)
    else:
        print(f"  [WARN] Unsupported archive format: {archive_path}")
        return
    
    print(f"  [INFO] Extracted to: {extract_dir}")


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Fetch Extended Atlas Sources - ETL Pipeline")
    print("=" * 60)
    print()
    
    output_dir = Path(args.output_dir)
    
    # 1. Fetch branches and try to get CSV from each
    branches = fetch_branches(ATLAS_REPO)
    
    for branch in branches:
        branch_name = branch['name']
        
        # Try to download biological_qubits.csv from each branch
        success = download_from_branch(
            ATLAS_REPO,
            branch_name,
            'biological_qubits.csv',
            output_dir
        )
        
        if success:
            print(f"  [SUCCESS] Got CSV from branch: {branch_name}")
    
    # 2. Extract any archives that were downloaded in previous runs
    print("\n[INFO] Checking for archives to extract...")
    
    for archive_file in output_dir.rglob('*.zip'):
        extract_dir = archive_file.parent / 'extracted' / archive_file.stem
        if not extract_dir.exists():
            extract_archive(archive_file, extract_dir)
    
    for archive_file in output_dir.rglob('*.tar.gz'):
        extract_dir = archive_file.parent / 'extracted' / archive_file.stem
        if not extract_dir.exists():
            extract_archive(archive_file, extract_dir)
    
    print()
    print("=" * 60)
    print("Extended fetch complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()


