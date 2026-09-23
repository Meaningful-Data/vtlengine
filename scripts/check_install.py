"""Check that the installed vtlengine works.

Imports it, prints the versions of its main dependencies and runs one statement on the
pandas and DuckDB engines. Two callers:

* ``scripts/check_micropip_install.mjs`` runs this file inside stock Pyodide right after
  installing the freshly built wasm wheel with micropip;
* the ``pip install vtlengine`` stage of ``.github/workflows/pyodide_test.yml`` runs it on
  the runner after installing the latest release from PyPI, with ``--latest`` to also check
  that pip installed the newest release PyPI advertises instead of falling back to an
  older one.

Raises SystemExit with a message on the first failed check (exit code 1), so the Pyodide
caller sees it as the last line of the traceback; returns normally when everything passes.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import sys
import urllib.request
from typing import Any

import pandas

from vtlengine import run

USAGE = "usage: python scripts/check_install.py [--latest]"

# Imported explicitly so that a dependency wheel that does not load shows up here, with its
# compiled part where the top-level package is pure Python (lxml).
DEPENDENCY_MODULES = (
    "duckdb",
    "lxml.etree",
    "networkx",
    "numpy",
    "pandas",
    "pyarrow",
    "pysdmx",
    "sqlglot",
)
DISTRIBUTIONS = (
    "vtlengine",
    "pysdmx",
    "lxml",
    "pandas",
    "numpy",
    "pyarrow",
    "duckdb",
    "networkx",
    "sqlglot",
)

DATA_STRUCTURES: dict[str, Any] = {
    "datasets": [
        {
            "name": "DS_1",
            "DataStructure": [
                {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
            ],
        }
    ]
}


def import_dependencies() -> None:
    """Import the dependencies that matter most, compiled parts included."""
    for name in DEPENDENCY_MODULES:
        importlib.import_module(name)


def versions_line() -> str:
    """One line with the installed versions of vtlengine and those dependencies."""
    return " | ".join(f"{name} {importlib.metadata.version(name)}" for name in DISTRIBUTIONS)


def check_run(use_duckdb: bool) -> None:
    """Run ``DS_r <- DS_1 * 10;`` on one engine and compare the result."""
    datapoints: dict[str, Any] = {"DS_1": pandas.DataFrame({"Id_1": [1, 2], "Me_1": [10.0, 20.0]})}
    result = run(
        script="DS_r <- DS_1 * 10;",
        data_structures=DATA_STRUCTURES,
        datapoints=datapoints,
        use_duckdb=use_duckdb,
    )
    data = getattr(result["DS_r"], "data", None)
    if not isinstance(data, pandas.DataFrame):
        raise SystemExit(f"run() with use_duckdb={use_duckdb}: DS_r has no data ({data!r})")
    got = data.sort_values("Id_1")["Me_1"].tolist()
    if got != [100.0, 200.0]:
        raise SystemExit(f"run() with use_duckdb={use_duckdb}: expected [100.0, 200.0], got {got}")
    print(f"run() with use_duckdb={use_duckdb}: OK")


def check_latest_release() -> None:
    """Fail unless the installed vtlengine is the latest release PyPI advertises."""
    installed = importlib.metadata.version("vtlengine")
    with urllib.request.urlopen("https://pypi.org/pypi/vtlengine/json", timeout=30) as response:
        latest = json.load(response)["info"]["version"]
    if installed != latest:
        raise SystemExit(
            f"vtlengine {installed} is installed but the latest release on PyPI is {latest}: "
            "pip did not install the latest release on this interpreter"
        )
    print(f"vtlengine {installed} is the latest release on PyPI")


def main(argv: list[str]) -> None:
    """Run the checks; ``--latest`` adds the PyPI comparison."""
    if argv not in ([], ["--latest"]):
        print(USAGE, file=sys.stderr)
        raise SystemExit(2)
    import_dependencies()
    print("versions:", versions_line())
    for use_duckdb in (False, True):
        check_run(use_duckdb)
    if argv:
        check_latest_release()


if __name__ == "__main__":
    main(sys.argv[1:])
