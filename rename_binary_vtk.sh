#!/bin/bash

# Find all files starting with "binary_" and rename them
find /home/jacopo/Documents/phd_research/Liggghts_simulations/cluster_simulations/ -type f -name "binary_*" | while read file; do
    # Extract the directory and new filename
    dir=$(dirname "$file")
    new_name=$(basename "$file" | sed 's/^binary_//')
    
    # Rename the file
    mv "$file" "$dir/$new_name"
    echo "Renamed: $file -> $dir/$new_name"
done
