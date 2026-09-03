"""Smoke test for the WebAssembly (Pyodide) wheel.

cibuildwheel runs this inside a ``pyodide venv`` (Node.js) right after installing the
freshly built ``pyemscripten`` wheel (see ``[tool.cibuildwheel.pyodide]`` in
pyproject.toml). ``tests/API`` cannot run wholesale under wasm -- no threads or
multiprocessing, no real filesystem or S3 -- so the release job only checks that the
wheel imports and that both execution engines run a script end to end, against the
pandas/numpy/pyarrow/duckdb wheels of the Pyodide distribution.

It is plain Python, so it also runs natively:  python scripts/wasm_smoke_test.py
"""

import pandas as pd

from vtlengine import run, semantic_analysis

SCRIPT = "DS_r <- DS_1 * 10;"
DATA_STRUCTURES = {
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


def main() -> None:
    structures = semantic_analysis(script=SCRIPT, data_structures=DATA_STRUCTURES)
    if list(structures) != ["DS_r"]:
        raise SystemExit(f"semantic_analysis returned {list(structures)!r}, expected ['DS_r']")
    print("semantic_analysis OK")

    for use_duckdb in (False, True):
        datapoints = {"DS_1": pd.DataFrame({"Id_1": [1, 2, 3], "Me_1": [1.5, None, 3.0]})}
        result = run(
            script=SCRIPT,
            data_structures=DATA_STRUCTURES,
            datapoints=datapoints,
            use_duckdb=use_duckdb,
        )
        values = result["DS_r"].data.sort_values("Id_1")["Me_1"].tolist()
        if not (
            len(values) == 3 and values[0] == 15.0 and pd.isna(values[1]) and values[2] == 30.0
        ):
            raise SystemExit(f"run(use_duckdb={use_duckdb}) returned {values!r}")
        print(f"run(use_duckdb={use_duckdb}) OK: {values}")

    print(f"vtlengine wasm smoke test passed (pandas {pd.__version__})")


if __name__ == "__main__":
    main()
