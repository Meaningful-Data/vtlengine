#!/usr/bin/env bash
#
# Build a self-contained JupyterLite site that runs the VTL Engine entirely in
# the browser (Pyodide kernel), with vtlengine + DuckDB + deps preloaded so
# `import vtlengine` works with no %pip / piplite step.
#
# Prerequisites (see README.md):
#   * A Python 3.10+ environment with the build tools:  pip install -r requirements.txt
#   * Node.js (used by the Pyodide kernel at build time)
#   * The vtlengine WebAssembly wheel for Pyodide 314.x (PEP 783
#     pyemscripten_2026_0_wasm32 ABI). Build it with ./build-wheel.sh and pass it
#     via VTLENGINE_WHEEL=..., or drop it into ./wheels/ beforehand.
#
# Usage:
#   VTLENGINE_WHEEL=/path/to/vtlengine-...-pyemscripten_2026_0_wasm32.whl ./build.sh
#
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# Keep in step with the xbuildenv build-wheel.sh targets: the 314.x distribution
# already ships pandas 3, numpy, pyarrow, duckdb 1.5.1, lxml, msgspec, networkx,
# jsonschema and httpx, so only vtlengine and its pure-Python deps are injected.
PYODIDE_VERSION="314.0.6"

WORK="${HERE}/.build"
WHEELS="${HERE}/wheels"
OUT="${HERE}/_output"
PYODIDE_TARBALL="${WORK}/pyodide-${PYODIDE_VERSION}.tar.bz2"
PY="${PYTHON:-python}"

mkdir -p "$WORK" "$WHEELS"

echo "==> 1/5  vtlengine wheel"
if [ -n "${VTLENGINE_WHEEL:-}" ]; then
    # build-wheel.sh (and CI) may already drop the wheel into $WHEELS; skip the
    # copy when VTLENGINE_WHEEL already points there (cp errors on same-file).
    dest="$WHEELS/$(basename "$VTLENGINE_WHEEL")"
    [ "$VTLENGINE_WHEEL" -ef "$dest" ] || cp "$VTLENGINE_WHEEL" "$WHEELS/"
    # A stale wheel from an earlier build would be injected too: keep only this one.
    find "$WHEELS" -name 'vtlengine-*.whl' ! -name "$(basename "$VTLENGINE_WHEEL")" -delete
fi
if [ "$(ls "$WHEELS"/vtlengine-*pyemscripten_2026_0_wasm32.whl 2>/dev/null | wc -l)" -ne 1 ]; then
    echo "ERROR: expected exactly one vtlengine wheel in $WHEELS. Build it with ./build-wheel.sh and set VTLENGINE_WHEEL." >&2
    exit 1
fi

echo "==> 2/5  pure-Python deps not bundled in Pyodide (the versions poetry.lock pins)"
# pysdmx's lxml >= 6.1.0 floor is not checked here: the served lockfile carries no
# version constraints, so the demo runs on the distribution's lxml 6.0.2 (see
# README.md). A next Pyodide release fixes this: pyodide/pyodide-recipes#656.
# Drop whatever an earlier build left (other versions, the duckdb wheel of the
# pre-314 flow...): every wheel in $WHEELS ends up in the served lockfile.
find "$WHEELS" -name '*.whl' ! -name 'vtlengine-*' -delete
"$PY" -m pip download --no-deps --quiet --dest "$WHEELS" \
    parsy==2.2 pysdmx==1.19.0 sdmxschemas==1.1.0 sqlglot==22.5.0 xmltodict==1.0.4

echo "==> 3/5  jupyter lite build (stock Pyodide ${PYODIDE_VERSION})"
[ -f "$PYODIDE_TARBALL" ] || curl -fsSL -o "$PYODIDE_TARBALL" \
    "https://github.com/pyodide/pyodide/releases/download/${PYODIDE_VERSION}/pyodide-${PYODIDE_VERSION}.tar.bz2"
# Also drop the doit state of the previous build: with the output gone but the state
# kept, `jupyter lite build` skips the tasks that patch jupyter-lite.json (appName and
# the kernel's pyodideUrl), and the served site then loads Pyodide from the kernel's
# default CDN instead of static/pyodide, where `import vtlengine` does not exist.
rm -rf "$OUT" "$HERE/.jupyterlite.doit.db"
( cd "$HERE" && jupyter lite build --pyodide="$PYODIDE_TARBALL" --contents=content --output-dir="$OUT" )
# Fail loudly if the kernel is not pointed at the bundled Pyodide.
"$PY" - "$OUT/jupyter-lite.json" <<'PY'
import json, sys

cfg = json.load(open(sys.argv[1]))["jupyter-config-data"]
kernel = cfg.get("litePluginSettings", {}).get("@jupyterlite/pyodide-kernel-extension:kernel", {})
url = kernel.get("pyodideUrl", "")
if "static/pyodide/pyodide" not in url:
    sys.exit(f"ERROR: jupyter-lite.json does not point the kernel at static/pyodide (pyodideUrl={url!r})")
print(f"  kernel pyodideUrl: {url}")
PY

echo "==> 4/5  inject wheels + patch the served lockfile (zero-install auto-load)"
cp "$WHEELS"/*.whl "$OUT/static/pyodide/"
"$PY" "${HERE}/patch_lock.py" "$OUT/static/pyodide"

echo "==> 5/5  redirect the demo root (/) to the vtl-demo notebook"
# Point the bare demo URL (e.g. /jupyterlite/) straight at the vtl-demo notebook.
# IMPORTANT: do NOT overwrite index.html. JupyterLite's config-utils.js fetches the
# site-root index.html and reads its embedded <script id="jupyter-config-data">, so
# replacing it with a bare stub makes the app crash on boot (textContent of null).
# Inject a <meta refresh> into its <head> instead; the relative target keeps working
# under any deploy path.
"$PY" - "$OUT/index.html" <<'PY'
import sys, pathlib

path = pathlib.Path(sys.argv[1])
html = path.read_text()
tag = '<meta http-equiv="refresh" content="0; url=lab/index.html?path=vtl-demo.ipynb">'
if tag in html:
    print("  redirect already present")
elif "<head>" in html:
    path.write_text(html.replace("<head>", "<head>\n    " + tag, 1))
    print("  injected redirect into <head>")
else:
    path.write_text(tag + "\n" + html)
    print("  prepended redirect (no <head> found)")
PY

echo
echo "Done. Serve the demo with:"
echo "    python -m http.server -d \"$OUT\" 8000"
echo "then open http://localhost:8000/  (redirects to the vtl-demo notebook)"
