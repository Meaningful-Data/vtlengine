#!/usr/bin/env python3
"""Configure which versions to build in documentation based on tag analysis."""

import re
import sys
from pathlib import Path
from typing import Optional

from version_utils import (
    find_latest_rc_tag,
    get_all_version_tags,
    get_latest_stable_versions,
    parse_version,
)


def should_build_rc_tags(
    tags: list[str], latest_stable_versions: list[str]
) -> tuple[bool, Optional[str]]:
    """
    Determine if rc tags should be built.

    Args:
        tags: All available version tags
        latest_stable_versions: List of latest stable versions

    Returns:
        Tuple of (should_build, latest_rc_tag)
        - should_build: True if latest rc is newer than latest stable
        - latest_rc_tag: The latest rc tag, or None
    """
    latest_rc = find_latest_rc_tag(tags)

    if not latest_rc:
        return (False, None)

    if not latest_stable_versions:
        # Only rc tags exist, build them
        return (True, latest_rc)

    # Compare versions - RC should be built if it's for a newer base version
    stable_base = parse_version(latest_stable_versions[0])[:3]
    rc_base = parse_version(latest_rc)[:3]

    return (rc_base > stable_base, latest_rc)


def generate_tag_whitelist(
    stable_versions: list[str], build_rc: bool, latest_rc: Optional[str]
) -> str:
    """
    Generate the tag whitelist regex pattern.

    Args:
        stable_versions: List of stable versions to include
        build_rc: Whether to build rc tags
        latest_rc: The latest rc tag (if any)

    Returns:
        Regex pattern string
    """
    if not stable_versions and not build_rc:
        return r"^v\d+\.\d+\.\d+$"

    patterns = []

    for version in stable_versions:
        patterns.append(f"{re.escape(version)}$")

    if build_rc and latest_rc:
        patterns.append(f"{re.escape(latest_rc)}$")

    if not patterns:
        return r"^v\d+\.\d+\.\d+$"

    return f"^({'|'.join(patterns)})"


def update_sphinx_config(tag_whitelist: str) -> None:
    """
    Update the Sphinx configuration file with the new tag whitelist.

    Args:
        tag_whitelist: The regex pattern for tag whitelist
    """
    conf_path = Path(__file__).parent.parent / "conf.py"

    if not conf_path.exists():
        print(f"Error: Configuration file not found: {conf_path}")
        sys.exit(1)

    content = conf_path.read_text(encoding="utf-8")
    pattern = r'smv_tag_whitelist = r"[^"]*"'

    if not re.search(pattern, content):
        print("Error: Could not find smv_tag_whitelist in conf.py")
        sys.exit(1)

    new_content = re.sub(pattern, f'smv_tag_whitelist = r"{tag_whitelist}"', content)

    if new_content == content:
        print(f"smv_tag_whitelist already set to: {tag_whitelist}")
    else:
        conf_path.write_text(new_content, encoding="utf-8")
        print(f"Updated smv_tag_whitelist to: {tag_whitelist}")


def main() -> int:
    """Main entry point."""
    print("Analyzing version tags...")

    all_tags = get_all_version_tags()
    stable_versions = get_latest_stable_versions(all_tags, limit=5)
    print(f"Latest stable versions (limit 5): {', '.join(stable_versions)}")

    build_rc, latest_rc = should_build_rc_tags(all_tags, stable_versions)

    if build_rc:
        print(f"Building rc tags: Latest rc ({latest_rc}) is the newest version")
    elif latest_rc:
        print(f"Skipping rc tags: Stable version exists that is same or newer than {latest_rc}")
    else:
        print("No rc tags found")

    tag_whitelist = generate_tag_whitelist(stable_versions, build_rc, latest_rc)
    print(f"Generated tag whitelist: {tag_whitelist}")

    update_sphinx_config(tag_whitelist)
    print("Sphinx configuration updated successfully")

    return 0


if __name__ == "__main__":
    sys.exit(main())
