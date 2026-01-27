#!/usr/bin/env python3
"""Generate root index.html that redirects to the latest stable documentation version."""

import re
import sys
from pathlib import Path
from typing import Optional


def parse_version(version_str: str) -> tuple[int, int, int, str]:
    """
    Parse a version string into a sortable tuple.

    Args:
        version_str: Version string like 'v1.5.0' or 'v1.5.0rc6'

    Returns:
        Tuple of (major, minor, patch, suffix) for sorting
    """
    # Remove 'v' prefix if present
    version_str = version_str.lstrip("v")

    # Split version and suffix (e.g., "1.5.0rc6" -> "1.5.0" and "rc6")
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)(.*)$", version_str)
    if not match:
        raise ValueError(f"Invalid version format: {version_str}")

    major = int(match.group(1))
    minor = int(match.group(2))
    patch = int(match.group(3))
    suffix = match.group(4)

    return (major, minor, patch, suffix)


def is_stable_version(version_str: str) -> bool:
    """Check if a version is stable (not pre-release)."""
    # Stable versions don't contain 'rc', 'alpha', 'beta', etc.
    return (
        "rc" not in version_str.lower()
        and "alpha" not in version_str.lower()
        and "beta" not in version_str.lower()
    )


def find_latest_stable_version(site_dir: Path) -> Optional[str]:
    """
    Find the latest stable version from built documentation directories.

    Args:
        site_dir: Path to the _site directory containing version subdirectories

    Returns:
        Latest stable version string (e.g., 'v1.5.0') or None if no stable versions found
    """
    # Find all directories matching version pattern
    version_dirs = []
    for item in site_dir.iterdir():
        if item.is_dir() and item.name.startswith("v"):
            try:
                # Validate it's a proper version
                parse_version(item.name)
                version_dirs.append(item.name)
            except ValueError:
                # Skip directories that don't match version pattern
                continue

    if not version_dirs:
        return None

    # Filter to stable versions only
    stable_versions = [v for v in version_dirs if is_stable_version(v)]

    if not stable_versions:
        # No stable versions, fall back to latest pre-release
        print("Warning: No stable versions found, using latest pre-release")
        stable_versions = version_dirs

    # Sort by semantic version
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
    # Determine site directory (default to _site)
    site_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "_site")

    if not site_dir.exists():
        print(f"Error: Site directory not found: {site_dir}")
        return 1

    # Find latest stable version
    latest_version = find_latest_stable_version(site_dir)

    if not latest_version:
        print("Error: No versions found in site directory")
        return 1

    print(f"Latest stable version: {latest_version}")

    # Generate redirect HTML
    redirect_html = generate_redirect_html(latest_version)

    # Write to index.html in site root
    index_path = site_dir / "index.html"
    index_path.write_text(redirect_html, encoding="utf-8")

    print(f"Generated redirect at {index_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
