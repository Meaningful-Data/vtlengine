#!/usr/bin/env python3
"""Rename the latest stable version directory to 'latest' and leave a redirect behind."""

import shutil
import sys
from pathlib import Path
from typing import Optional

from version_utils import is_stable_version, parse_version


def find_latest_stable_version(site_dir: Path) -> Optional[str]:
    """
    Find the latest stable version from built documentation directories.

    Args:
        site_dir: Path to the _site directory containing version subdirectories

    Returns:
        Latest stable version string (e.g., 'v1.5.0') or None if no stable versions found
    """
    version_dirs = []
    for item in site_dir.iterdir():
        if item.is_dir() and item.name.startswith("v"):
            try:
                parse_version(item.name)
                version_dirs.append(item.name)
            except ValueError:
                continue

    if not version_dirs:
        return None

    stable_versions = [v for v in version_dirs if is_stable_version(v)]

    if not stable_versions:
        print("Warning: No stable versions found, using latest pre-release")
        stable_versions = version_dirs

    stable_versions.sort(key=parse_version, reverse=True)
    return stable_versions[0]


def generate_redirect_html(target: str) -> str:
    """Generate a minimal HTML redirect page."""
    return f"""<!DOCTYPE html>
<html><head><meta http-equiv="refresh" content="0; url=../{target}/index.html"></head>
<body><a href="../{target}/index.html">Redirect</a></body></html>
"""


def main() -> int:
    """Main entry point for the script."""
    site_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "_site")

    if not site_dir.exists():
        print(f"Error: Site directory not found: {site_dir}")
        return 1

    latest_version = find_latest_stable_version(site_dir)

    if not latest_version:
        print("Error: No versions found in site directory")
        return 1

    source_dir = site_dir / latest_version
    latest_dir = site_dir / "latest"

    if latest_dir.exists():
        shutil.rmtree(latest_dir)

    # Move instead of copy — no duplication
    source_dir.rename(latest_dir)

    # Leave a redirect at the old version path so existing links still work
    source_dir.mkdir()
    (source_dir / "index.html").write_text(generate_redirect_html("latest"), encoding="utf-8")

    print(f"Moved {latest_version} -> latest (redirect left at {source_dir})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
