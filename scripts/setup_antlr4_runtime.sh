#!/bin/bash
# Downloads and extracts the ANTLR4 C++ runtime source for building the C++ parser extension.
# Run this once before building: ./scripts/setup_antlr4_runtime.sh

set -euo pipefail

ANTLR4_VERSION="4.13.1"
TARGET_DIR="third_party/antlr4-cpp-runtime"
TARBALL_URL="https://github.com/antlr/antlr4/archive/refs/tags/${ANTLR4_VERSION}.tar.gz"

if [ -d "${TARGET_DIR}/src" ]; then
    echo "ANTLR4 C++ runtime already exists at ${TARGET_DIR}/src"
    exit 0
fi

echo "Downloading ANTLR4 ${ANTLR4_VERSION} C++ runtime..."
TMPDIR=$(mktemp -d)
curl -sL "${TARBALL_URL}" -o "${TMPDIR}/antlr4.tar.gz"

echo "Extracting runtime sources..."
mkdir -p "${TARGET_DIR}"
tar -xzf "${TMPDIR}/antlr4.tar.gz" \
    --strip-components=4 \
    -C "${TARGET_DIR}" \
    "antlr4-${ANTLR4_VERSION}/runtime/Cpp/runtime/src"

rm -rf "${TMPDIR}"

echo "Done. ANTLR4 C++ runtime extracted to ${TARGET_DIR}/src/"
