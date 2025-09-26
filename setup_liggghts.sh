#!/bin/bash
set -e pipefail

# Configurable version
LIGGGHTS_COMMIT="bbd23c85"  
REPO_URL="https://github.com/CFDEMproject/LIGGGHTS-PUBLIC.git"
PATCH_FILE="../liggghts_patch.patch"  # Relative to liggghts_source

# Clone or update to commit
if [ -d "liggghts_source" ]; then
    echo "liggghts_source exists. Checking out ${LIGGGHTS_COMMIT}..."
    cd liggghts_source
    git fetch origin
    git checkout "${LIGGGHTS_COMMIT}"
    cd ..
else
    echo "Cloning LIGGGHTS-PUBLIC at ${LIGGGHTS_COMMIT}..."
    git clone "${REPO_URL}" liggghts_source
    cd liggghts_source
    git checkout "${LIGGGHTS_COMMIT}"
    cd ..
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