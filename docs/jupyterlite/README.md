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
2. runs `jupyter lite build` against stock Pyodide 314.0.6;
3. resolves the wheel's dependencies with **micropip itself**, in that served Pyodide
   (`resolve_wheels.mjs`, Node.js): the same resolution a user gets from
   `micropip.install("vtlengine")` — the packages the distribution ships from its
   lockfile, the rest (`pysdmx`, `sqlglot`, `parsy`, `sdmxschemas`, `xmltodict` today)
   from PyPI at the newest releases the engine allows — then **merges
   `micropip.freeze()` into the served `pyodide-lock.json`** (`patch_lock.py`, wheels
   downloaded next to it) so Pyodide auto-loads everything on `import` — the key to the
   zero-install experience;
4. prunes the served Pyodide distribution to what the demo can reach
   (`prune_dist.py`): the dependency closure of `vtlengine` and of the kernel,
   ~50 MB of the ~380 MB the tarball ships.

Everything else (`pandas` 3, `numpy`, `pyarrow`, `duckdb` 1.5.1, `lxml`, `msgspec`,
`networkx`, `jsonschema`, `httpx`) already ships in Pyodide 314.

## Build

Prerequisites: Node.js 20+, and a Python 3.10+ environment
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
  (the vtlengine wheel), `.build/` (the Pyodide tarball, re-downloaded when missing, and
  micropip's resolution) and `.cache/` (jupyterlite's extraction of that tarball,
  re-extracted when missing).
- `static/pyodide/` is pruned to what the demo can reach: 42 of the 362 packages of
  the distribution, ~50 MB instead of ~380 MB (`prune_dist.py`). Visitors download
  the same files either way, since Pyodide only fetches what a notebook imports;
  the pruning shrinks the Pages artifact, of which the demo was ~90%. The cost:
  `%pip install` of a *compiled* package outside that closure (scipy, polars...)
  no longer works in the demo. Pure-Python packages still resolve from PyPI.
- Pyodide is single-threaded; the DuckDB engine (`use_duckdb=True`) runs on an in-memory
  database there, so no spill-to-disk or remote file access is involved.
- `pysdmx` comes in at the newest release the engine allows, micropip's pick, and runs on
  the `lxml` 6.0.2 of the Pyodide 314 distribution (built against libxml2 2.9.10 and
  libxslt 1.1.33). pysdmx
  accepts that since 1.20.0: its floor is `lxml >= 6.0.2` on Emscripten and `lxml >= 6.1.0`
  everywhere else (a security floor: lxml 6.1.0 fixes CVE-2026-41066 and bundles patched
  libxml2/libxslt; pysdmx's XML validation disables external entity resolution explicitly,
  so the CVE fix does not depend on the lxml version, see
  <https://github.com/bis-med-it/pysdmx/pull/692>). The same allowance is what lets a plain
  `micropip.install("vtlengine")` resolve on stock Pyodide 314. It is temporary: once a
  Pyodide release ships lxml 6.1 (<https://github.com/pyodide/pyodide-recipes/pull/656>
  moves the recipes to lxml 6.1.3, libxslt 1.1.45 and libxml2 2.15.3), the floor goes back
  to `lxml >= 6.1.0` everywhere. The patched lockfile carries no version constraints, so
  nothing in the build checks the floor the resolved pysdmx declares.
- `scripts/check_micropip_install.mjs` performs that plain install for the wheel
  `pyodide_test.yml` (weekly, and on pull requests that touch `pyproject.toml` or the
  check itself) and `release.yml` have just built, on stock Pyodide in Node.js: micropip
  resolution against the Pyodide lockfile and PyPI, then `scripts/check_install.py`
  (`import vtlengine` and one statement on both engines). Run it locally with
  `npm install --no-save pyodide@314.0.6` and `node scripts/check_micropip_install.mjs <wheel>`.
  The weekly run also checks that the latest release on PyPI installs with
  `pip install vtlengine` on every supported Python and OS, then runs the same script with
  `--latest`.
- The served lockfile differs from a plain micropip install in three deliberate ways, all
  in `patch_lock.py`, because JupyterLite loads packages per `import` where
  `micropip.install` loads the whole set at once: `pysdmx` gains `pandas` and `pyarrow`
  (its `data` extra, which vtlengine brings itself, so a notebook may import `pysdmx` before
  `vtlengine`); `networkx` loses the `matplotlib` dependency Pyodide's recipe declares
  (networkx 3.x needs none of it, and it is ~10 MB); and `httpx` gains `certifi`, which
  stock Pyodide ships but never auto-loads, so remote SDMX URLs work out of the box.
