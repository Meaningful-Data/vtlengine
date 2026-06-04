"""Add vtlengine and its non-base dependencies to a Pyodide lockfile so that
``import vtlengine`` auto-loads them in JupyterLite with no ``%pip``/``piplite``
step.

Run *after* ``jupyter lite build``, pointing at the served Pyodide directory
(the matching wheels must already be copied into it)::

    python patch_lock.py <output>/static/pyodide

``imports`` drives JupyterLite's import-triggered auto-load; ``depends`` mirrors
the runtime dependency graph resolved by ``micropip`` (regenerate with
``micropip.freeze()`` if vtlengine's dependencies change). Everything else
(pandas, numpy, pyarrow, lxml, msgspec, networkx, jsonschema, httpx) already
ships in the stock Pyodide distribution.
"""

import hashlib
import json
import sys
from pathlib import Path

EXTRA = {
    "vtlengine": {
        "imports": ["vtlengine"],
        "depends": [
            "duckdb",
            "jsonschema",
            "networkx",
            "numpy",
            "pandas",
            "pyarrow",
            "pysdmx",
            "sqlglot",
        ],
    },
    "duckdb": {"imports": ["duckdb"], "depends": []},
    "pysdmx": {
        "imports": ["pysdmx"],
        # ssl + certifi ship in stock Pyodide but are NOT auto-loaded: httpx imports
        # them at runtime for HTTPS, so without them a remote read_sdmx/run_sdmx(URL)
        # fails with "No module named 'ssl'". Listing them here loads them alongside
        # pysdmx so remote SDMX URLs work out of the box (ssl also pulls libopenssl).
        # Local-file SDMX needs neither.
        "depends": [
            "certifi",
            "httpx",
            "lxml",
            "msgspec",
            "parsy",
            "sdmxschemas",
            "ssl",
            "xmltodict",
        ],
    },
    "parsy": {"imports": ["parsy"], "depends": []},
    "sdmxschemas": {"imports": ["sdmxschemas"], "depends": []},
    "sqlglot": {"imports": ["sqlglot"], "depends": []},
    "xmltodict": {"imports": ["xmltodict"], "depends": []},
}


def find_wheel(dist: Path, pkg: str) -> Path:
    for pattern in (f"{pkg.replace('-', '_')}-*.whl", f"{pkg}-*.whl"):
        hits = sorted(dist.glob(pattern))
        if hits:
            return hits[0]
    raise SystemExit(f"wheel for {pkg!r} not found in {dist}")


def main() -> None:
    dist = Path(sys.argv[1])
    lock_path = dist / "pyodide-lock.json"
    lock = json.loads(lock_path.read_text())

    for pkg, meta in EXTRA.items():
        whl = find_wheel(dist, pkg)
        version = whl.name.split("-")[1]
        lock["packages"][pkg] = {
            "name": pkg,
            "version": version,
            "file_name": whl.name,
            "install_dir": "site",
            "package_type": "package",
            "sha256": hashlib.sha256(whl.read_bytes()).hexdigest(),
            "imports": meta["imports"],
            "depends": meta["depends"],
            "unvendored_tests": False,
        }
        print(f"  + {pkg} {version}")

    lock_path.write_text(json.dumps(lock))
    print(f"patched {lock_path} ({len(lock['packages'])} packages)")


if __name__ == "__main__":
    main()
