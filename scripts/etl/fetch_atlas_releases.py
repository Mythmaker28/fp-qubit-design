#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch ALL releases from biological-qubits-atlas (including pre-releases).

Uses GitHub API to:
1. List all releases
2. Download CSV/TSV/JSON assets
3. Log provenance (tag, date, SHA256)
"""

import argparse
import json
import hashlib
from pathlib import Path
from datetime import datetime
import urllib.request
import urllib.error


ATLAS_REPO = "Mythmaker28/biological-qubits-atlas"
GITHUB_API_BASE = "https://api.github.com"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch all Atlas releases"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/atlas/releases",
        help="Output directory for releases"
    )
    parser.add_argument(
        "--include-prerelease",
        action="store_true",
        default=True,
        help="Include pre-releases"
    )
    return parser.parse_args()


def fetch_releases(repo: str, include_prerelease: bool = True) -> list:
    """Fetch all releases from GitHub API."""
    url = f"{GITHUB_API_BASE}/repos/{repo}/releases"
    
    print(f"[INFO] Fetching releases from: {url}")
    
    try:
        with urllib.request.urlopen(url) as response:
            releases = json.loads(response.read())
    except urllib.error.HTTPError as e:
        print(f"[ERROR] HTTP {e.code}: {e.reason}")
        return []
    
    if not include_prerelease:
        releases = [r for r in releases if not r.get('prerelease', False)]
    
    print(f"[INFO] Found {len(releases)} releases")
    
    return releases


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for block in iter(lambda: f.read(4096), b''):
            sha256.update(block)
    return sha256.hexdigest()


def download_asset(asset: dict, output_dir: Path) -> dict:
    """Download a single asset."""
    asset_name = asset['name']
    asset_url = asset['browser_download_url']
    asset_size = asset['size']
    
    output_file = output_dir / asset_name
    
    print(f"  [INFO] Downloading: {asset_name} ({asset_size} bytes)")
    
    try:
        urllib.request.urlretrieve(asset_url, output_file)
    except Exception as e:
        print(f"  [ERROR] Failed to download {asset_name}: {e}")
        return None
    
    # Compute checksum
    sha256 = compute_sha256(output_file)
    
    return {
        'name': asset_name,
        'size': asset_size,
        'sha256': sha256,
        'url': asset_url,
        'downloaded_at': datetime.now().isoformat(),
    }


def download_release_assets(release: dict, base_output_dir: Path) -> dict:
    """Download all tabular assets (CSV/TSV/JSON) from a release."""
    tag = release['tag_name']
    published_at = release.get('published_at', 'unknown')
    prerelease = release.get('prerelease', False)
    
    print(f"\n[INFO] Processing release: {tag} (published: {published_at}, prerelease: {prerelease})")
    
    # Create release directory
    release_dir = base_output_dir / tag
    release_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter tabular assets
    assets = release.get('assets', [])
    tabular_assets = [
        a for a in assets
        if any(a['name'].lower().endswith(ext) for ext in ['.csv', '.tsv', '.json'])
    ]
    
    if not tabular_assets:
        print(f"  [WARN] No tabular assets found in {tag}")
        return None
    
    print(f"  [INFO] Found {len(tabular_assets)} tabular assets")
    
    # Download assets
    downloaded = []
    for asset in tabular_assets:
        result = download_asset(asset, release_dir)
        if result:
            downloaded.append(result)
    
    return {
        'tag': tag,
        'published_at': published_at,
        'prerelease': prerelease,
        'assets': downloaded,
    }


def generate_harvest_log(harvest_results: list, output_file: Path):
    """Generate API_HARVEST_LOG.md."""
    lines = []
    lines.append("# API HARVEST LOG - Biological Qubits Atlas")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Repository**: {ATLAS_REPO}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    total_releases = len(harvest_results)
    total_assets = sum(len(r['assets']) for r in harvest_results if r)
    
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total releases**: {total_releases}")
    lines.append(f"- **Total assets downloaded**: {total_assets}")
    lines.append("")
    
    lines.append("## Releases")
    lines.append("")
    
    for result in harvest_results:
        if not result:
            continue
        
        lines.append(f"### {result['tag']}")
        lines.append("")
        lines.append(f"- **Published**: {result['published_at']}")
        lines.append(f"- **Prerelease**: {result['prerelease']}")
        lines.append(f"- **Assets**: {len(result['assets'])}")
        lines.append("")
        
        if result['assets']:
            lines.append("| Asset | Size (bytes) | SHA256 |")
            lines.append("|-------|--------------|--------|")
            for asset in result['assets']:
                lines.append(f"| `{asset['name']}` | {asset['size']} | `{asset['sha256'][:16]}...` |")
            lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append("**License**: Data from Biological Qubits Atlas is licensed under CC BY 4.0")
    lines.append("")
    lines.append("**Citation**:")
    lines.append("```")
    lines.append("Lepesteur, T. (2025). Biological Qubits Atlas. GitHub.")
    lines.append("https://github.com/Mythmaker28/biological-qubits-atlas")
    lines.append("```")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"\n[INFO] Harvest log saved: {output_file}")


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Fetch Atlas Releases - ETL Pipeline")
    print("=" * 60)
    print()
    
    # Fetch releases
    releases = fetch_releases(ATLAS_REPO, include_prerelease=args.include_prerelease)
    
    if not releases:
        print("[ERROR] No releases found!")
        return
    
    # Download assets
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    harvest_results = []
    for release in releases:
        result = download_release_assets(release, base_output_dir)
        if result:
            harvest_results.append(result)
    
    # Generate log
    log_file = Path("reports/API_HARVEST_LOG.md")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    generate_harvest_log(harvest_results, log_file)
    
    print()
    print("=" * 60)
    print(f"Harvest complete! {len(harvest_results)} releases processed")
    print("=" * 60)


if __name__ == "__main__":
    main()


