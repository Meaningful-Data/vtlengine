# VTL Engine — JupyterLite demo

A self-contained [JupyterLite](https://jupyterlite.readthedocs.io) site that runs
the VTL Engine **entirely in the browser** on the Pyodide kernel — no server, no
backend. Opening the demo notebook and running `import vtlengine` auto-loads the
WebAssembly parser and the DuckDB execution engine; there is **no `%pip install`
step**.

## How it works

`vtlengine`'s parser is a compiled C++/pybind11 extension, so it needs a
WebAssembly build. The build therefore:

1. compiles `vtlengine` to a `pyemscripten_2026_0_wasm32` wasm wheel (PEP 783,
   the ABI of the Pyodide 314.x line — see `build-wheel.sh`);
2. gathers the pure-Python deps not bundled in Pyodide (`pysdmx`, `sdmxschemas`,
   `parsy`, `xmltodict`, `sqlglot`), at the versions `poetry.lock` pins;
3. runs `jupyter lite build` against stock Pyodide 314.0.6, then **adds these
   wheels to the served `pyodide-lock.json`** (`patch_lock.py`) so Pyodide
   auto-loads them on `import` — the key to the zero-install experience;
4. prunes the served Pyodide distribution to what the demo can reach
   (`prune_dist.py`): the dependency closure of `vtlengine` and of the kernel,
   ~60 MB of the ~380 MB the tarball ships.

Everything else (`pandas` 3, `numpy`, `pyarrow`, `duckdb` 1.5.1, `lxml`, `msgspec`,
`networkx`, `jsonschema`, `httpx`) already ships in Pyodide 314.

## Build

Prerequisites: Node.js, and a Python 3.10+ environment
(`pip install -r requirements.txt`). Versions are pinned to the JupyterLite
0.8.x line, whose Pyodide kernel is 314.x — the ABI the wasm wheel targets.

```bash
# 1. Build the vtlengine wasm wheel (needs emsdk 5.0.3 + pyodide-build/xbuildenv
#    314.0.6 on a Python 3.14 host — see the header of build-wheel.sh).
./build-wheel.sh

# 2. Assemble the JupyterLite site (downloads the deps + Pyodide, builds,
#    patches the lockfile, prunes the distribution).
./build.sh
```

`build.sh` also accepts a prebuilt wheel via `VTLENGINE_WHEEL=/path/to/wheel` —
for instance the `cp314-pyodide_wasm32` wheel `release.yml` builds with
cibuildwheel and publishes to PyPI.

## Run locally

```bash
python -m http.server -d _output 8000
```

Open <http://localhost:8000/lab/index.html> and run `content/vtl-demo.ipynb`.

## Deployment

The `build-jupyterlite` job in `.github/workflows/docs.yml` runs both steps in CI
(host Python 3.14 + emsdk 5.0.3, with the wasm wheel and Emscripten SDK cached)
and the docs `build` job publishes the result at `/jupyterlite/` on the docs
site — e.g. <https://docs.vtlengine.meaningfuldata.eu/jupyterlite/lab/index.html>.
The docs workflow runs on releases, manual dispatch, or a merged `cr-N` PR whose
issue carries the `documentation` label.

## Notes

- Build artifacts are git-ignored and safe to delete: `_output/` (the site), `wheels/`
  (the injected wheels), `.build/` (the Pyodide tarball, re-downloaded when missing)
  and `.cache/` (jupyterlite's extraction of that tarball, re-extracted when missing).
- `static/pyodide/` is pruned to what the demo can reach: 51 of the 362 packages of
  the distribution, ~60 MB instead of ~380 MB (`prune_dist.py`). Visitors download
  the same files either way, since Pyodide only fetches what a notebook imports;
  the pruning shrinks the Pages artifact, of which the demo was ~90%. The cost:
  `%pip install` of a *compiled* package outside that closure (scipy, polars...)
  no longer works in the demo. Pure-Python packages still resolve from PyPI.
- Pyodide is single-threaded; `run()` uses in-memory DuckDB, so no spill-to-disk
  or remote file access is involved.
- `pysdmx` is injected at the version `poetry.lock` pins. Its `lxml >= 6.1.0` floor
  (a security floor: lxml 6.1.0 fixes CVE-2026-41066 and bundles patched
  libxml2/libxslt) is not enforced by the patched lockfile, so the demo runs it on
  the `lxml` 6.0.2 of the Pyodide 314 distribution, built against libxml2 2.9.10 and
  libxslt 1.1.33. For the same reason a plain `micropip.install("vtlengine")` on
  stock Pyodide 314 fails on `lxml>=6.1.0`; installing `pysdmx[xml]==1.16.0` first
  and `vtlengine` in a second call gets through, on that same lxml. The floor is
  being addressed on the Pyodide side: <https://github.com/pyodide/pyodide-recipes/pull/656>
  moves the recipes to lxml 6.1.3, libxslt 1.1.45 and libxml2 2.15.3, so a next
  Pyodide release ships a compliant `lxml` and both workarounds become unnecessary.
- To refresh the dependency graph baked into `patch_lock.py`, re-run
  `micropip.freeze()` in the target Pyodide and update the `EXTRA` table.
