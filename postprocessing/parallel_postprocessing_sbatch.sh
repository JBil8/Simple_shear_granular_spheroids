#!/bin/bash
set -euo pipefail

CONFIG_FILE="../config.yaml"

# Read config with envsubst (so $USER etc. expand)
CONFIG=$(envsubst < "$CONFIG_FILE")

# Parse YAML values
aspectRatios=($(yq -r '.aspect_ratios[]' <<< "$CONFIG"))
Is=($(yq -r '.inertial_numbers[]' <<< "$CONFIG"))
cofs=($(yq -r '.cofs[]' <<< "$CONFIG"))
max_parallel_tasks=$(yq -r '.max_parallel_tasks' <<< "$CONFIG")

# Slurm parameters
ntasks=$(yq -r '.ntasks' <<< "$CONFIG")
cpus_per_task=$(yq -r '.cpus_per_task' <<< "$CONFIG")
time=$(yq -r '.time' <<< "$CONFIG")

echo "Start of the loop"

# Loop through params
for cof in "${cofs[@]}"; do
    for ap in "${aspectRatios[@]}"; do
        if (( $(echo "$ap < 1" | bc -l) )); then
            raw_pressure=$(echo "50 * $ap" | bc -l)
            # Format to one decimal place and remove trailing .0
            pressure=$(echo "$raw_pressure" | awk '{printf "%.1f", $1}' | sed 's/\.0$//')
        else
            pressure=50
        fi
        
        for I in "${Is[@]}"; do
            sbatch <<EOL
# !/bin/bash
# SBATCH -n $ntasks
# SBATCH --ntasks=$ntasks
# SBATCH --cpus-per-task=$cpus_per_task
# SBATCH -t $time
# SBATCH -o output_post_ap_${ap}_cof_${cof}_I_${I}_%j.txt
# SBATCH -e error_post_ap_${ap}_cof_${cof}_I_${I}_%j.txt

python main.py -c $cof -a $ap -v $I -s $pressure
EOL

            # Parallel limit logic
            running_tasks=$(jobs -p | wc -l)
            while [ $running_tasks -ge $max_parallel_tasks ]; do
                sleep 1
                running_tasks=$(jobs -p | wc -l)
            done
        done
    done
done

wait
echo "All post-processing jobs submitted to Slurm."
