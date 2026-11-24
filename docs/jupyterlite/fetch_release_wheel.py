#!/usr/bin/env python3
"""Download the wasm wheel of the newest vtlengine release from PyPI.

The newest release is the highest version that has a ``pyemscripten_2026_0_wasm32`` wheel,
release candidates included. The wheel is written to the folder given as the only argument,
and the other vtlengine wheels in it are removed, as build.sh expects exactly one. Every error
(PyPI unreachable, no release with the wheel, a wrong digest) exits with a non-zero status.
"""

import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any, Optional

PROJECT = "vtlengine"
PLATFORM = "pyemscripten_2026_0_wasm32"
VERSIONS_TO_TRY = 5
TIMEOUT = 60


def newest_versions() -> list[str]:
    """The versions of the project on PyPI, newest first, pre-releases included."""
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "pip", "index", "versions", PROJECT, "--pre"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for line in out.splitlines():
        if line.startswith("Available versions:"):
            return [version.strip() for version in line.split(":", 1)[1].split(",")]
    raise SystemExit(f"Could not read the versions of {PROJECT} from pip:\n{out}")


def wasm_wheel(version: str) -> Optional[dict[str, Any]]:
    """The PyPI entry of the wasm wheel of a version, or None when it has none."""
    url = f"https://pypi.org/pypi/{PROJECT}/{version}/json"
    with urllib.request.urlopen(url, timeout=TIMEOUT) as response:  # noqa: S310
        files: list[dict[str, Any]] = json.load(response)["urls"]
    return next((f for f in files if f["filename"].endswith(f"{PLATFORM}.whl")), None)


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        raise SystemExit(f"Usage: {Path(sys.argv[0]).name} <folder for the wheel>")
    folder = Path(argv[0])
    folder.mkdir(parents=True, exist_ok=True)

    for version in newest_versions()[:VERSIONS_TO_TRY]:
        entry = wasm_wheel(version)
        if entry is None:
            print(f"{PROJECT} {version} has no {PLATFORM} wheel, trying the previous version")
            continue
        with urllib.request.urlopen(entry["url"], timeout=TIMEOUT) as response:  # noqa: S310
            content = response.read()
        if hashlib.sha256(content).hexdigest() != entry["digests"]["sha256"]:
            raise SystemExit(f"The sha256 of {entry['filename']} does not match PyPI's")
        for old in folder.glob(f"{PROJECT}-*.whl"):
            old.unlink()
        (folder / entry["filename"]).write_bytes(content)
        print(f"{PROJECT} {version}: {folder / entry['filename']}")
        return 0
    raise SystemExit(f"None of the newest {VERSIONS_TO_TRY} versions has a {PLATFORM} wheel")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
