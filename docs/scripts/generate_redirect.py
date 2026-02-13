#!/usr/bin/env python3
"""Generate root index.html that redirects to the latest stable documentation version."""

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


def generate_redirect_html(target_version: str) -> str:
    """
    Generate HTML content that redirects to the target version.

    Args:
        target_version: Version to redirect to (e.g., 'v1.5.0')

    Returns:
        HTML content as string
    """
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="0; url=./{target_version}/">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Redirecting to VTL Engine Documentation</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                         "Helvetica Neue", Arial, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            background-color: #f5f5f5;
        }}
        .container {{
            text-align: center;
            padding: 2rem;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        a {{
            color: #2980b9;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>VTL Engine Documentation</h1>
        <p>Redirecting to <a href="./{target_version}/">version {target_version}</a>...</p>
        <p><small>If you are not redirected automatically, please click the link above.</small></p>
    </div>
</body>
</html>
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

    print(f"Latest stable version: {latest_version}")

    redirect_html = generate_redirect_html("latest")
    index_path = site_dir / "index.html"
    index_path.write_text(redirect_html, encoding="utf-8")

    print(f"Generated redirect to 'latest' (currently {latest_version}) at {index_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
