#!/usr/bin/env bash
#
# Build the vtlengine WebAssembly wheel for the Pyodide 0.29.3 / pyodide_2025_0
# ABI (the one JupyterLite's pyodide kernel 0.7.x uses), and drop it in ./wheels.
#
# Prerequisites:
#   * Host Python 3.13 — pyodide-build requires the host interpreter's
#     major.minor to match the target Pyodide's Python, and 0.29.3 targets 3.13.
#       uv python install 3.13   # (or any 3.13 interpreter)
#   * pyodide-build with the 0.29.3 cross-build environment:
#       pip install pyodide-build && pyodide xbuildenv install 0.29.3
#   * Emscripten 4.0.9 active (matches Pyodide 0.29.3):
#       git clone https://github.com/emscripten-core/emsdk && cd emsdk
#       ./emsdk install 4.0.9 && ./emsdk activate 4.0.9 && source ./emsdk_env.sh
#
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$(cd "$(dirname "$0")" && pwd)/wheels"
mkdir -p "$OUT"

command -v emcc >/dev/null || { echo "emcc not found — activate emsdk 4.0.9 (source emsdk_env.sh)" >&2; exit 1; }
command -v pyodide >/dev/null || { echo "pyodide-build not found — pip install pyodide-build" >&2; exit 1; }

bash "$REPO/scripts/setup_antlr4_runtime.sh"

( cd "$REPO" && rm -rf dist && pyodide build )

# pyodide-build stamps the wheel `pyemscripten_2025_0`; the 0.29.3 runtime and
# the duckdb-pyodide wheels use the `pyodide_2025_0` platform tag. Retag to match.
WHEEL="$(ls "$REPO"/dist/vtlengine-*-pyemscripten_2025_0_wasm32.whl)"
python -m wheel tags --platform-tag pyodide_2025_0_wasm32 --remove "$WHEEL" >/dev/null
cp "$REPO"/dist/vtlengine-*-pyodide_2025_0_wasm32.whl "$OUT/"
echo "vtlengine wheel -> $(ls "$OUT"/vtlengine-*-pyodide_2025_0_wasm32.whl)"
