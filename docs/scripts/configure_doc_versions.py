#!/usr/bin/env python3
"""Configure which versions to build in documentation based on tag analysis."""

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

from version_utils import (
    find_latest_rc_tag,
    get_all_version_tags,
    get_latest_stable_versions,
    parse_version,
)

SMV_WHITELIST_PATH = Path(__file__).parent.parent / "_smv_whitelist.json"


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


def get_current_branch() -> Optional[str]:
    """Get the current git branch name, or None if in detached HEAD state."""
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            check=True,
        )
        branch = result.stdout.strip()
        return branch if branch else None
    except subprocess.CalledProcessError:
        return None


def write_whitelist_config(tag_whitelist: str, include_current_branch: bool = False) -> None:
    """
    Write the sphinx-multiversion whitelist configuration to a JSON file.

    Args:
        tag_whitelist: The regex pattern for tag whitelist
        include_current_branch: Whether to add the current git branch to smv_branch_whitelist
    """
    branch_whitelist = r"^main$"

    if include_current_branch:
        current_branch = get_current_branch()
        if current_branch and current_branch != "main":
            branch_whitelist = f"^(main|{re.escape(current_branch)})$"
            print(f"Updated smv_branch_whitelist to include: {current_branch}")

    config = {
        "smv_tag_whitelist": tag_whitelist,
        "smv_branch_whitelist": branch_whitelist,
    }

    SMV_WHITELIST_PATH.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote whitelist config to {SMV_WHITELIST_PATH.name}")


def main() -> int:
    """Main entry point."""
    include_current_branch = "--include-current-branch" in sys.argv

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

    write_whitelist_config(tag_whitelist, include_current_branch=include_current_branch)
    print("Configuration updated successfully")

    return 0


if __name__ == "__main__":
    sys.exit(main())
