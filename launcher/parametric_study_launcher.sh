#!/bin/bash
set -euo pipefail

# Config file
CONFIG_FILE="config.yaml"

# Load config variables using yq
SIM_DATA_DIR=$(yq -r '.sim_data_dir' $CONFIG_FILE)
executable=$(yq -r '.executable' $CONFIG_FILE)
input_script=$(yq -r '.input_script' $CONFIG_FILE)
radius=$(yq -r '.radius' $CONFIG_FILE)
density=$(yq -r '.density' $CONFIG_FILE)
aspectRatios=($(yq -r '.aspect_ratios[]' $CONFIG_FILE))
inertialNumbers=($(yq -r '.inertial_numbers[]' $CONFIG_FILE))
cofs=($(yq -r '.cofs[]' $CONFIG_FILE))

# Slurm parameters
ntasks=$(yq -r '.ntasks' $CONFIG_FILE)
cpus_per_task=$(yq -r '.cpus_per_task' $CONFIG_FILE)
mem=$(yq -r '.mem' $CONFIG_FILE)
time=$(yq -r '.time' $CONFIG_FILE)
mail_user=$(yq -r '.mail_user' $CONFIG_FILE)
mail_type=$(yq -r '.mail_type' $CONFIG_FILE)

echo "Start of the loop"

# Loop through the values of COF, aspectRatio, and I
for COF in "${cofs[@]}"; do
    for aspectRatio in "${aspectRatios[@]}"; do
        for I in "${inertialNumbers[@]}"; do    
            # Job directory under SIM_DATA_DIR
            job_dir="${SIM_DATA_DIR}/job_cof_${COF}_aspectRatio_${aspectRatio}_$(date +%Y%m%d%H%M%S)"
            mkdir -p "$job_dir"

            job_script="$job_dir/job_${COF}_${aspectRatio}.sh"
            cat << EOF > "$job_script"
#!/bin/bash
#SBATCH --job-name=_alpha_${aspectRatio}_s_${ntasks}_cof_${COF}
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --ntasks=${ntasks}
#SBATCH --cpus-per-task=${cpus_per_task}
#SBATCH --mem=${mem}
#SBATCH --time=${time}
#SBATCH --mail-type=${mail_type}
#SBATCH --mail-user=${mail_user}

srun $executable -v density $density -v aspectRatio $aspectRatio -v COF $COF -v Radius $radius -v I $I -in $SIM_DATA_DIR/$input_script
EOF

            chmod +x "$job_script"
            sbatch "$job_script"
        done
    done
done

echo "All jobs submitted to Slurm."
