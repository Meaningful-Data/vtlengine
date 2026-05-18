#!/usr/bin/env python3
"""Configure which versions to build in documentation based on tag analysis."""

import json
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

VERSIONS_JSON_PATH = Path(__file__).parent.parent / "_versions.json"


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


def get_current_branch() -> Optional[str]:
    """Get the current git branch name, or None if in detached HEAD state."""
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],  # noqa: S603, S607
            capture_output=True,
            text=True,
            check=True,
        )
        branch = result.stdout.strip()
        return branch if branch else None
    except subprocess.CalledProcessError:
        return None


def build_versions_list(
    stable_versions: list[str],
    latest_rc: Optional[str],
    build_rc: bool,
    extra_branch: Optional[str],
) -> list[str]:
    """
    Assemble the ordered list of refs to build.

    Order (most-recent first for display):
      main, <extra_branch?>, <latest_rc if build_rc>, *stable_versions

    The list deduplicates entries so an explicit `--include-current-branch=main`
    does not duplicate the always-present `main` entry.
    """
    versions = ["main"]
    if extra_branch and extra_branch not in versions:
        versions.append(extra_branch)
    if build_rc and latest_rc:
        versions.append(latest_rc)
    versions.extend(stable_versions)
    return versions


def write_versions_config(versions: list[str], latest_stable: Optional[str]) -> None:
    """Write the resolved version list to docs/_versions.json."""
    config = {
        "versions": versions,
        "latest_stable": latest_stable,
    }

    VERSIONS_JSON_PATH.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote version list to {VERSIONS_JSON_PATH.name}")


def main() -> int:
    """Main entry point."""
    include_current_branch = "--include-current-branch" in sys.argv

    print("Analyzing version tags...")

    all_tags = get_all_version_tags()
    stable_versions = get_latest_stable_versions(all_tags, limit=5)
    print(f"Latest stable versions (limit 5): {', '.join(stable_versions) or '(none)'}")

    build_rc, latest_rc = should_build_rc_tags(all_tags, stable_versions)

    if build_rc:
        print(f"Building rc tags: Latest rc ({latest_rc}) is the newest version")
    elif latest_rc:
        print(f"Skipping rc tags: Stable version exists that is same or newer than {latest_rc}")
    else:
        print("No rc tags found")

    extra_branch: Optional[str] = None
    if include_current_branch:
        current_branch = get_current_branch()
        if current_branch and current_branch != "main":
            extra_branch = current_branch
            print(f"Including current branch in build list: {current_branch}")

    latest_stable = stable_versions[0] if stable_versions else None
    versions = build_versions_list(stable_versions, latest_rc, build_rc, extra_branch)
    print(f"Resolved versions to build: {', '.join(versions)}")
    print(f"Latest stable: {latest_stable or '(none)'}")

    write_versions_config(versions, latest_stable)
    print("Configuration updated successfully")

    return 0


if __name__ == "__main__":
    sys.exit(main())
