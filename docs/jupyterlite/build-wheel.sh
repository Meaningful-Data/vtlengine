#!/usr/bin/env bash
#
# Build the vtlengine WebAssembly wheel for Pyodide 314.x (Python 3.14, the PEP 783
# `pyemscripten_2026_0_wasm32` ABI JupyterLite's pyodide kernel 0.8.x uses), and
# drop it in ./wheels.
#
# Prerequisites:
#   * Host Python 3.14 — pyodide-build requires the host interpreter's
#     major.minor to match the target Pyodide's Python, and 314.x targets 3.14.
#       (use actions/setup-python, uv, pyenv, or any system Python 3.14)
#   * pyodide-build with the 314.0.6 cross-build environment:
#       pip install pyodide-build && pyodide xbuildenv install 314.0.6
#   * Emscripten 5.0.3 active (the version Pyodide 314.x pins — check with
#     `pyodide xbuildenv search --all`):
#       git clone https://github.com/emscripten-core/emsdk && cd emsdk
#       ./emsdk install 5.0.3 && ./emsdk activate 5.0.3 && source ./emsdk_env.sh
#
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$(cd "$(dirname "$0")" && pwd)/wheels"
mkdir -p "$OUT"

command -v emcc >/dev/null || { echo "emcc not found — activate emsdk 5.0.3 (source emsdk_env.sh)" >&2; exit 1; }
command -v pyodide >/dev/null || { echo "pyodide-build not found — pip install pyodide-build" >&2; exit 1; }

bash "$REPO/scripts/setup_antlr4_runtime.sh"

( cd "$REPO" && rm -rf dist && pyodide build )

# pyodide-build stamps the wheel with the PEP 783 `pyemscripten_2026_0_wasm32` tag,
# which is exactly what Pyodide 314's micropip expects (no retag step any more).
rm -f "$OUT"/vtlengine-*.whl
cp "$REPO"/dist/vtlengine-*-pyemscripten_2026_0_wasm32.whl "$OUT/"
echo "vtlengine wheel -> $(ls "$OUT"/vtlengine-*-pyemscripten_2026_0_wasm32.whl)"
