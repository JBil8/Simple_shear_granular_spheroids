#!/bin/bash

# # Define the parameters
#aspectRatio=(1.0 1.2 1.5 1.8 2.0 2.5 3.0)
cofs=(0.0 0.4 1.0)
#Is=(0.0316 0.01 0.00316 0.001 0.000316 0.0001)
aspectRatio=(3.0)
cofs=(0.0)
Is=(0.000316)

# Define the maximum number of parallel tasks
max_parallel_tasks=30

echo "Start of the loop"

# Loop through the values of -var flag for COF
for cof in "${cofs[@]}"
do
    # Loop through the values of -var flag for ap
    for ap in "${aspectRatio[@]}"
    do
        # Loop through the values of -var flag for phi
        for I in "${Is[@]}"
        do
            # Submit the job directly to sbatch
            sbatch <<EOL
#!/bin/bash
#SBATCH -n 1 #Request 1 task (core)
#SBATCH --ntasks=1                      # Number of tasks (processes)
#SBATCH --cpus-per-task=32              # Number of CPU cores per task
#SBATCH -t 0-00:05 #Request runtime of 1 hour
##SBATCH -o output_post_%j.txt #redirect output to output_post_JOBID.txt
##SBATCH -e error_post_%j.txt #redirect errors to error_post_JOBID.txt
python main_shear_les.py -c $cof -a $ap -v $I -s 50
EOL

            # Limit the number of parallel tasks
            running_tasks=$(jobs -p | wc -l)
            while [ $running_tasks -ge $max_parallel_tasks ]; do
                sleep 1
                running_tasks=$(jobs -p | wc -l)
            done
        done
    done
done

# Wait for all background jobs to finish
wait

echo "All post-processing jobs submitted to Slurm."
