#!/bin/bash
# Setup environment for compiling/running LIGGGHTS
# Reads module list from modules_used.txt

MODULE_FILE="../modules_used.txt"
if [[ ! -f "$MODULE_FILE" ]]; then
  echo "ERROR: $MODULE_FILE not found"
  exit 1
fi

while IFS= read -r mod; do
  [[ -z "$mod" ]] && continue   # skip empty lines
  echo "Loading $mod..."
  if ! module load "$mod" 2>/dev/null; then
    echo "WARNING: Module $mod not found"
  fi
done < "$MODULE_FILE"

# Install yq if not present (idempotent: skips if already there)
if ! command -v yq &> /dev/null; then
    echo "Installing yq..."
    wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /tmp/yq
    chmod +x /tmp/yq
    export PATH="/tmp:$PATH"  # Make it available for the script
fi

echo "Environment setup complete."

