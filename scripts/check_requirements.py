"""Check the installed packages against the requirements declared from vtlengine down.

Used by ``.github/workflows/ubuntu_test_26_04.yml``, which installs the dependencies with APT
(the few that Ubuntu does not package with ``pip install --no-dependencies``) and vtlengine
itself with ``--no-deps``, so pip never compares the installed versions with the declared
ranges. This walks the requirements from vtlengine down, with the extras each package asks
for (``pysdmx[xml]``), and prints the requirements of vtlengine and of the packages pip
installed, each with the installed version.

Any of those that the installed version does not satisfy fails the check. Requirements
declared by APT packages (under ``/usr/lib/python3/dist-packages``) are Ubuntu's: they are
printed only when unmet, and ignored, because the archive does not always agree with its own
metadata (Ubuntu 26.04 ships httpcore 1.0.9, which requires h11>=0.16, next to h11 0.14.0).
That is also why a whole-environment ``pip check`` cannot be used.

Raises SystemExit (exit code 1) after listing every requirement not met; returns normally when
all of them are.
"""

from __future__ import annotations

from collections import deque
from importlib.metadata import Distribution, PackageNotFoundError, distribution
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = "vtlengine"
APT_SITE_PACKAGES = Path("/usr/lib/python3/dist-packages")


def installed_by_apt(dist: Distribution) -> bool:
    """Whether a distribution lives where APT installs Python packages."""
    return Path(str(dist.locate_file(""))).resolve() == APT_SITE_PACKAGES


def applicable_requirements(dist: Distribution, extras: frozenset[str]) -> list[Requirement]:
    """The requirements of a distribution that apply on this interpreter with these extras."""
    applicable = []
    for line in dist.requires or []:
        requirement = Requirement(line)
        marker = requirement.marker
        if marker is None or any(marker.evaluate({"extra": e}) for e in ("", *sorted(extras))):
            applicable.append(requirement)
    return applicable


def describe(requirement: Requirement) -> str:
    """The requirement without its marker, e.g. ``pysdmx[xml]<2.0,>=1.15.1``."""
    extras = f"[{','.join(sorted(requirement.extras))}]" if requirement.extras else ""
    return f"{requirement.name}{extras}{requirement.specifier}"


def main() -> None:
    """Walk the requirements from vtlengine down and fail on the unmet ones it enforces."""
    pending: deque[tuple[str, frozenset[str]]] = deque([(ROOT, frozenset())])
    visited: set[tuple[str, frozenset[str]]] = set()
    # A package asked for with and without extras is walked twice: report its base
    # requirements once.
    reported: set[tuple[str, str]] = set()
    unmet = 0
    while pending:
        name, extras = pending.popleft()
        if (canonicalize_name(name), extras) in visited:
            continue
        visited.add((canonicalize_name(name), extras))
        dist = distribution(name)
        enforced = not installed_by_apt(dist)
        for requirement in applicable_requirements(dist, extras):
            if (canonicalize_name(name), str(requirement)) in reported:
                continue
            reported.add((canonicalize_name(name), str(requirement)))
            try:
                installed = distribution(requirement.name).version
            except PackageNotFoundError:
                status, installed = "MISSING", "not installed"
            else:
                pending.append((requirement.name, frozenset(requirement.extras)))
                satisfied = requirement.specifier.contains(installed, prereleases=True)
                status = "OK" if satisfied else "CONFLICT"
            if status == "OK" and not enforced:
                continue
            note = "" if enforced or status == "OK" else "  (declared by an APT package: ignored)"
            if status != "OK" and enforced:
                unmet += 1
            parent = f"{dist.metadata['Name']} {dist.version}"
            print(f"{status:<8} {parent} requires {describe(requirement)}: {installed}{note}")
    if unmet:
        raise SystemExit(f"{unmet} requirement(s) not satisfied by the installed packages")
    print("All the requirements checked are satisfied by the installed packages")


if __name__ == "__main__":
    main()
