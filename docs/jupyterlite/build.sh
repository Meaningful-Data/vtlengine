#!/usr/bin/env bash
#
# Build a self-contained JupyterLite site that runs the VTL Engine entirely in
# the browser (Pyodide kernel), with vtlengine + DuckDB + deps preloaded so
# `import vtlengine` works with no %pip / piplite step.
#
# Prerequisites (see README.md):
#   * A Python 3.13 environment with the build tools:  pip install -r requirements.txt
#   * Node.js (used by the Pyodide kernel at build time)
#   * The vtlengine WebAssembly wheel for the Pyodide 0.29.3 / pyodide_2025_0 ABI.
#     Build it with ./build-wheel.sh and pass it via VTLENGINE_WHEEL=...,
#     or drop it into ./wheels/ beforehand.
#
# Usage:
#   VTLENGINE_WHEEL=/path/to/vtlengine-...-pyodide_2025_0_wasm32.whl ./build.sh
#
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PYODIDE_VERSION="0.29.3"
DUCKDB_WHEEL_URL="https://github.com/xlwings/duckdb-pyodide/releases/download/duckdb-v1.5.0-pyodide-0.29.3/duckdb-1.5.0-cp313-cp313-pyodide_2025_0_wasm32.whl"

WORK="${HERE}/.build"
WHEELS="${HERE}/wheels"
OUT="${HERE}/_output"
PYODIDE_TARBALL="${WORK}/pyodide-${PYODIDE_VERSION}.tar.bz2"
PY="${PYTHON:-python}"

mkdir -p "$WORK" "$WHEELS"

echo "==> 1/5  vtlengine wheel"
if [ -n "${VTLENGINE_WHEEL:-}" ]; then
    cp "$VTLENGINE_WHEEL" "$WHEELS/"
fi
if ! ls "$WHEELS"/vtlengine-*pyodide_2025_0_wasm32.whl >/dev/null 2>&1; then
    echo "ERROR: no vtlengine wheel in $WHEELS. Build it with ./build-wheel.sh and set VTLENGINE_WHEEL." >&2
    exit 1
fi

echo "==> 2/5  DuckDB wasm wheel (>=1.4, from duckdb-pyodide)"
[ -f "$WHEELS/$(basename "$DUCKDB_WHEEL_URL")" ] || curl -fsSL -o "$WHEELS/$(basename "$DUCKDB_WHEEL_URL")" "$DUCKDB_WHEEL_URL"

echo "==> 3/5  pure-Python deps not bundled in Pyodide"
"$PY" -m pip download --no-deps --quiet --dest "$WHEELS" \
    parsy==2.2 pysdmx==1.15.1 sdmxschemas==1.0.0 sqlglot==30.8.0 xmltodict==1.0.4

echo "==> 4/5  jupyter lite build (stock Pyodide ${PYODIDE_VERSION})"
[ -f "$PYODIDE_TARBALL" ] || curl -fsSL -o "$PYODIDE_TARBALL" \
    "https://github.com/pyodide/pyodide/releases/download/${PYODIDE_VERSION}/pyodide-${PYODIDE_VERSION}.tar.bz2"
rm -rf "$OUT"
( cd "$HERE" && jupyter lite build --pyodide="$PYODIDE_TARBALL" --contents=content --output-dir="$OUT" )

echo "==> 5/5  inject wheels + patch the served lockfile (zero-install auto-load)"
cp "$WHEELS"/*.whl "$OUT/static/pyodide/"
"$PY" "${HERE}/patch_lock.py" "$OUT/static/pyodide"

echo
echo "Done. Serve the demo with:"
echo "    python -m http.server -d \"$OUT\" 8000"
echo "then open http://localhost:8000/lab/index.html and run content/vtl-demo.ipynb"
