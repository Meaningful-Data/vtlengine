"""Shared utilities for version parsing and filtering across documentation scripts."""

import re
import subprocess
from typing import Optional


def _parse_suffix(suffix: str) -> tuple[int, int]:
    """
    Parse a version suffix into a sortable tuple.

    Stable versions (empty suffix) sort higher than pre-releases.
    Pre-release numbers are compared numerically (rc10 > rc9).

    Args:
        suffix: Version suffix like '', 'rc6', 'alpha1', 'beta2'

    Returns:
        Tuple of (is_stable, pre_release_number) for sorting.
        Stable: (1, 0), Pre-release: (0, N)
    """
    if not suffix:
        return (1, 0)
    match = re.search(r"(\d+)$", suffix)
    pre_num = int(match.group(1)) if match else 0
    return (0, pre_num)


def parse_version(version_str: str) -> tuple[int, int, int, int, int]:
    """
    Parse a version string into a sortable tuple.

    Args:
        version_str: Version string like 'v1.5.0', 'v1.5.0rc6', or 'v1.1'

    Returns:
        Tuple of (major, minor, patch, is_stable, pre_release_num) for sorting.
        Stable versions sort higher than pre-releases (e.g., 1.5.0 > 1.5.0rc10).
        Pre-release numbers are compared numerically (rc10 > rc9).

    Raises:
        ValueError: If the version string doesn't match expected format
    """
    # Remove 'v' prefix if present
    version_str = version_str.lstrip("v")

    # Try to match full version with patch (e.g., "1.5.0rc6" or "1.5.0")
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)(.*)$", version_str)
    if match:
        return (
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
            *_parse_suffix(match.group(4)),
        )

    # Try to match version without patch (e.g., "1.1")
    match = re.match(r"^(\d+)\.(\d+)(.*)$", version_str)
    if match:
        return (
            int(match.group(1)),
            int(match.group(2)),
            0,  # Default to 0 if no patch version
            *_parse_suffix(match.group(3)),
        )

    raise ValueError(f"Invalid version format: {version_str}")


def is_stable_version(version_str: str) -> bool:
    """
    Check if a version is stable (not pre-release).

    Args:
        version_str: Version string to check

    Returns:
        True if the version is stable (no rc, alpha, or beta suffix)
    """
    lower_version = version_str.lower()
    return (
        "rc" not in lower_version and "alpha" not in lower_version and "beta" not in lower_version
    )


def get_all_version_tags() -> list[str]:
    """
    Get all version tags from git.

    Returns:
        List of version tag strings (e.g., ['v1.0.0', 'v1.1.0', ...])
    """
    result = subprocess.run(
        ["git", "tag", "-l", "v*"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )
    return [tag.strip() for tag in result.stdout.strip().split("\n") if tag.strip()]


def find_latest_rc_tag(tags: list[str]) -> Optional[str]:
    """
    Find the latest release candidate tag.

    Args:
        tags: List of version tags

    Returns:
        The latest RC tag, or None if no RC tags exist
    """
    rc_tags = [tag for tag in tags if not is_stable_version(tag)]
    if not rc_tags:
        return None

    rc_tags.sort(key=parse_version, reverse=True)
    return rc_tags[0]


def find_latest_stable_tag(tags: list[str]) -> Optional[str]:
    """
    Find the latest stable tag.

    Args:
        tags: List of version tags

    Returns:
        The latest stable tag, or None if no stable tags exist
    """
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
        key = (parsed[0], parsed[1])
        if key not in version_groups:
            version_groups[key] = []
        version_groups[key].append(tag)

    # For each major.minor group, keep only the latest patch version
    latest_per_group = []
    for versions in version_groups.values():
        versions.sort(key=parse_version, reverse=True)
        latest_per_group.append(versions[0])

    # Sort all latest versions and return top N
    latest_per_group.sort(key=parse_version, reverse=True)
    return latest_per_group[:limit]
