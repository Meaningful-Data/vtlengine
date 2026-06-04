# VTL Engine — JupyterLite demo

A self-contained [JupyterLite](https://jupyterlite.readthedocs.io) site that runs
the VTL Engine **entirely in the browser** on the Pyodide kernel — no server, no
backend. Opening the demo notebook and running `import vtlengine` auto-loads the
WebAssembly parser and the DuckDB execution engine; there is **no `%pip install`
step**.

## How it works

`vtlengine`'s parser is a compiled C++/pybind11 extension, so it needs a
WebAssembly build, and its `duckdb` dependency (≥1.4) is not in the stock Pyodide
distribution. The build therefore:

1. compiles `vtlengine` to a `pyodide_2025_0` wasm wheel (see `build-wheel.sh`);
2. gathers `duckdb` 1.5.0 (from [duckdb-pyodide](https://github.com/xlwings/duckdb-pyodide))
   plus the pure-Python deps not bundled in Pyodide;
3. runs `jupyter lite build` against stock Pyodide 0.29.3, then **adds these
   wheels to the served `pyodide-lock.json`** (`patch_lock.py`) so Pyodide
   auto-loads them on `import` — the key to the zero-install experience.

Everything else (`pandas`, `numpy`, `pyarrow`, `lxml`, `msgspec`, `networkx`,
`jsonschema`, `httpx`) already ships in Pyodide 0.29.3.

## Build

Prerequisites: Node.js, and a Python 3.13 environment
(`pip install -r requirements.txt`). Versions are pinned to the JupyterLite
0.7.x line, whose Pyodide kernel is 0.29.3 — the ABI the wasm wheels target.

```bash
# 1. Build the vtlengine wasm wheel (needs emsdk 4.0.9 + pyodide-build/xbuildenv
#    0.29.3 on a Python 3.13 host — see the header of build-wheel.sh).
./build-wheel.sh

# 2. Assemble the JupyterLite site (downloads duckdb + deps + Pyodide, builds,
#    patches the lockfile).
./build.sh
```

`build.sh` also accepts a prebuilt wheel via `VTLENGINE_WHEEL=/path/to/wheel`.

## Run locally

```bash
python -m http.server -d _output 8000
```

Open <http://localhost:8000/lab/index.html> and run `content/vtl-demo.ipynb`.

## Deployment

The `build-jupyterlite` job in `.github/workflows/docs.yml` runs both steps in CI
(host Python 3.13 + emsdk 4.0.9, with the wasm wheel and Emscripten SDK cached)
and the docs `build` job publishes the result at `/jupyterlite/` on the docs
site — e.g. <https://docs.vtlengine.meaningfuldata.eu/jupyterlite/lab/index.html>.
The docs workflow runs on releases, manual dispatch, or a merged `cr-N` PR whose
issue carries the `documentation` label.

## Notes

- Build artifacts (`_output/`, `wheels/`, `.build/`) are git-ignored.
- Pyodide is single-threaded; `run()` uses in-memory DuckDB, so no spill-to-disk
  or remote file access is involved.
- To refresh the dependency graph baked into `patch_lock.py`, re-run
  `micropip.freeze()` in the target Pyodide and update the `EXTRA` table.
