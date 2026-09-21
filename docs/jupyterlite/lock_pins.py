"""Print the ``name==version`` pins of the wheels ``build.sh`` injects into the demo.

The names are the injected packages of ``patch_lock.EXTRA`` (every key but vtlengine)
and the versions are the ones ``poetry.lock`` resolves for the Python of the vtlengine
wasm wheel (``cp314`` -> 3.14), so the demo ships the dependency versions the engine is
tested with and the list cannot drift from the lock::

    python lock_pins.py <vtlengine wheel>

Needs Python 3.11+ (``tomllib``).
"""

import re
import sys
from pathlib import Path

import tomllib
from packaging.markers import Marker
from patch_lock import EXTRA

LOCK = Path(__file__).resolve().parents[2] / "poetry.lock"
INJECTED = [name for name in EXTRA if name != "vtlengine"]


def python_version(wheel: str) -> str:
    """``3.14`` for ``vtlengine-...-cp314-cp314-pyemscripten_2026_0_wasm32.whl``."""
    match = re.search(r"-cp3(\d+)-", Path(wheel).name)
    if match is None:
        raise SystemExit(f"ERROR: no cp3XX tag in the wheel name {Path(wheel).name!r}")
    return f"3.{match.group(1)}"


def marker_environment(version: str) -> dict[str, str]:
    """The PEP 508 environment of Pyodide on that Python, for the lock's markers."""
    return {
        "implementation_name": "cpython",
        "implementation_version": f"{version}.0",
        "os_name": "posix",
        "platform_machine": "wasm32",
        "platform_python_implementation": "CPython",
        "platform_release": "",
        "platform_system": "Emscripten",
        "platform_version": "",
        "python_full_version": f"{version}.0",
        "python_version": version,
        "sys_platform": "emscripten",
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python lock_pins.py <vtlengine wheel>")
    env = marker_environment(python_version(sys.argv[1]))
    lock = tomllib.loads(LOCK.read_text())
    pins = []
    for name in INJECTED:
        versions = sorted(
            {
                entry["version"]
                for entry in lock["package"]
                if entry["name"] == name
                and ("markers" not in entry or Marker(entry["markers"]).evaluate(env))
            }
        )
        if len(versions) != 1:
            raise SystemExit(
                f"ERROR: {LOCK} resolves {name} to {versions or 'nothing'} on Python "
                f"{env['python_version']}; expected exactly one version"
            )
        pins.append(f"{name}=={versions[0]}")
    print(" ".join(pins))


if __name__ == "__main__":
    main()
