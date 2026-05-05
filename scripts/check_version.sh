#!/bin/bash
# Checks that the version in pyproject.toml matches __version__ in src/vtlengine/__init__.py

set -euo pipefail

PYPROJECT_VERSION=$(grep -oP '^version\s*=\s*"\K[^"]+' pyproject.toml)
MODULE_VERSION=$(grep -oP '__version__\s*=\s*"\K[^"]+' src/vtlengine/__init__.py)

if [ "$PYPROJECT_VERSION" != "$MODULE_VERSION" ]; then
    echo "❌ Version mismatch detected!"
    echo "pyproject.toml version: $PYPROJECT_VERSION"
    echo "__version__ value: $MODULE_VERSION"
    exit 1
else
    echo "✅ Versions match ($PYPROJECT_VERSION)"
fi
