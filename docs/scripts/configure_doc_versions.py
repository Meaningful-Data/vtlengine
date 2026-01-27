#!/usr/bin/env python3
"""Configure which versions to build in documentation based on tag analysis."""

import re
import subprocess
import sys
from pathlib import Path
from typing import Optional


def parse_version(version_str: str) -> tuple[int, int, int, str]:
    """
    Parse a version string into a sortable tuple.

    Args:
        version_str: Version string like 'v1.5.0', 'v1.5.0rc6', or 'v1.1'

    Returns:
        Tuple of (major, minor, patch, suffix) for sorting
    """
    # Remove 'v' prefix if present
    version_str = version_str.lstrip("v")

    # Try to match full version with patch (e.g., "1.5.0rc6" or "1.5.0")
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)(.*)$", version_str)
    if match:
        major = int(match.group(1))
        minor = int(match.group(2))
        patch = int(match.group(3))
        suffix = match.group(4)
        return (major, minor, patch, suffix)

    # Try to match version without patch (e.g., "1.1")
    match = re.match(r"^(\d+)\.(\d+)(.*)$", version_str)
    if match:
        major = int(match.group(1))
        minor = int(match.group(2))
        patch = 0  # Default to 0 if no patch version
        suffix = match.group(3)
        return (major, minor, patch, suffix)

    raise ValueError(f"Invalid version format: {version_str}")


def is_stable_version(version_str: str) -> bool:
    """Check if a version is stable (not pre-release)."""
    return (
        "rc" not in version_str.lower()
        and "alpha" not in version_str.lower()
        and "beta" not in version_str.lower()
    )


def get_all_version_tags() -> list[str]:
    """Get all version tags from git."""
    result = subprocess.run(
        ["git", "tag", "-l", "v*"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )
    tags = [tag.strip() for tag in result.stdout.strip().split("\n") if tag.strip()]
    return tags


def find_latest_rc_tag(tags: list[str]) -> Optional[str]:
    """Find the latest rc tag."""
    rc_tags = [tag for tag in tags if not is_stable_version(tag)]
    if not rc_tags:
        return None

    rc_tags.sort(key=parse_version, reverse=True)
    return rc_tags[0]


def find_latest_stable_tag(tags: list[str]) -> Optional[str]:
    """Find the latest stable tag."""
    stable_tags = [tag for tag in tags if is_stable_version(tag)]
    if not stable_tags:
        return None

    stable_tags.sort(key=parse_version, reverse=True)
    return stable_tags[0]


def get_latest_stable_versions(tags: list[str], limit: int = 5) -> list[str]:
    """
    Get the latest N stable versions following semantic versioning.

    Only includes the highest patch version for each major.minor combination.
    For example, if we have v1.2.0, v1.2.1, v1.2.2, only v1.2.2 is included.

    Args:
        tags: List of all version tags
        limit: Maximum number of stable versions to return (default: 5)

    Returns:
        List of latest stable version tags, sorted newest first
    """
    stable_tags = [tag for tag in tags if is_stable_version(tag)]
    if not stable_tags:
        return []

    # Group by major.minor version
    version_groups: dict[tuple[int, int], list[str]] = {}
    for tag in stable_tags:
        parsed = parse_version(tag)
        major, minor = parsed[0], parsed[1]
        key = (major, minor)
        if key not in version_groups:
            version_groups[key] = []
        version_groups[key].append(tag)

    # For each major.minor group, keep only the latest patch version
    latest_per_group = []
    for versions in version_groups.values():
        versions.sort(key=parse_version, reverse=True)
        latest_per_group.append(versions[0])  # Highest patch version

    # Sort all latest versions and return top N
    latest_per_group.sort(key=parse_version, reverse=True)
    return latest_per_group[:limit]


def should_build_rc_tags(latest_stable_versions: list[str]) -> tuple[bool, Optional[str]]:
    """
    Determine if rc tags should be built.

    Args:
        latest_stable_versions: List of latest stable versions

    Returns:
        Tuple of (should_build, latest_rc_tag)
        - should_build: True if latest rc is newer than latest stable
        - latest_rc_tag: The latest rc tag, or None
    """
    tags = get_all_version_tags()
    latest_rc = find_latest_rc_tag(tags)

    if not latest_rc:
        # No rc tags at all
        return (False, None)

    if not latest_stable_versions:
        # Only rc tags exist, build them
        return (True, latest_rc)

    # Get the latest stable version
    latest_stable = latest_stable_versions[0]

    # Compare versions
    stable_version = parse_version(latest_stable)
    rc_version = parse_version(latest_rc)

    # RC tags have same version but with suffix, so we need to compare base version
    stable_base = stable_version[:3]  # (major, minor, patch)
    rc_base = rc_version[:3]

    if rc_base > stable_base:
        # RC is for a newer version than latest stable
        return (True, latest_rc)
    else:
        # Stable is same or newer version
        return (False, latest_rc)


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
        # Fallback: match all stable versions
        return r"^v\d+\.\d+\.\d+$"

    # Build a pattern that matches specific versions
    patterns = []

    # Add each stable version as an exact match
    for version in stable_versions:
        escaped_version = re.escape(version)
        patterns.append(f"{escaped_version}$")

    # Add rc if needed
    if build_rc and latest_rc:
        escaped_rc = re.escape(latest_rc)
        patterns.append(f"{escaped_rc}$")

    if not patterns:
        return r"^v\d+\.\d+\.\d+$"

    # Combine all patterns
    combined = "|".join(patterns)
    return f"^({combined})"


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

    # Replace the smv_tag_whitelist line
    # Use a function to avoid backslash interpretation issues
    def replace_whitelist(match: re.Match) -> str:  # type: ignore[type-arg]
        return f'smv_tag_whitelist = r"{tag_whitelist}"'

    pattern = r'smv_tag_whitelist = r"[^"]*"'

    # Check if pattern exists in content
    if not re.search(pattern, content):
        print("Error: Could not find smv_tag_whitelist in conf.py")
        sys.exit(1)

    new_content = re.sub(pattern, replace_whitelist, content)

    if new_content == content:
        print(f"smv_tag_whitelist already set to: {tag_whitelist}")
    else:
        conf_path.write_text(new_content, encoding="utf-8")
        print(f"Updated smv_tag_whitelist to: {tag_whitelist}")


def main() -> int:
    """Main entry point."""
    print("Analyzing version tags...")

    # Get all tags
    all_tags = get_all_version_tags()

    # Get latest 5 stable versions
    stable_versions = get_latest_stable_versions(all_tags, limit=5)
    print(f"Latest stable versions (limit 5): {', '.join(stable_versions)}")

    # Check if we should build rc
    build_rc, latest_rc = should_build_rc_tags(stable_versions)

    if build_rc:
        print(f"Building rc tags: Latest rc ({latest_rc}) is the newest version")
    else:
        if latest_rc:
            print(f"Skipping rc tags: Stable version exists that is same or newer than {latest_rc}")
        else:
            print("No rc tags found")

    # Generate whitelist
    tag_whitelist = generate_tag_whitelist(stable_versions, build_rc, latest_rc)
    print(f"Generated tag whitelist: {tag_whitelist}")

    update_sphinx_config(tag_whitelist)
    print("Sphinx configuration updated successfully")

    return 0


if __name__ == "__main__":
    sys.exit(main())
