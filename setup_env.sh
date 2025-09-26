#!/bin/bash
# Setup environment for compiling/running LIGGGHTS
# Reads module list from modules_used.txt

MODULE_FILE="modules_used.txt"

if [[ ! -f "$MODULE_FILE" ]]; then
    echo "ERROR: $MODULE_FILE not found!"
    exit 1
fi

# Extract all "name/version" tokens (skip headers like 'Currently Loaded Modules:')
REQUIRED_MODULES=$(grep -oE '[A-Za-z0-9._+-]+/[0-9][^ ]*' "$MODULE_FILE")

for mod in $REQUIRED_MODULES; do
    if module avail "$mod" 2>&1 | grep -q "$mod"; then
        echo "Loading $mod..."
        module load "$mod"
    else
        echo "WARNING: Module $mod not found on this system."
        echo "         You may need to load an equivalent manually."
    fi
done

echo "Environment setup complete."

