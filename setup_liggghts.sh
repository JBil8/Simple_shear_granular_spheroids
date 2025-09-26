#!/bin/bash
set -euo pipefail

# Configurable version
LIGGGHTS_VERSION="3.8.0"  # Pinned to specific release tag
REPO_URL="https://github.com/CFDEMproject/LIGGGHTS-PUBLIC.git"  # HTTPS for broader access
PATCH_FILE="../liggghts_patch.patch"  # Relative to liggghts_source

# Fetch and clone
if [ -d "liggghts_source" ]; then
    echo "liggghts_source already exists. Checking out ${LIGGGHTS_VERSION}..."
    cd liggghts_source
    git fetch origin
    git checkout "${LIGGGHTS_VERSION}"
    cd ..
else
    echo "Cloning LIGGGHTS-PUBLIC at ${LIGGGHTS_VERSION}..."
    git clone -b "${LIGGGHTS_VERSION}" "${REPO_URL}" liggghts_source
fi

# Apply patch
cd liggghts_source
if patch -p1 -N --dry-run < "${PATCH_FILE}" >/dev/null 2>&1; then
    echo "Applying custom patch..."
    patch -p1 < "${PATCH_FILE}"
else
    echo "Patch already applied or not needed."
fi

# Compile (load env first)
source ../setup_env.sh
bash ../compiler.sh

echo "LIGGGHTS setup complete. Executable at: build/liggghts"