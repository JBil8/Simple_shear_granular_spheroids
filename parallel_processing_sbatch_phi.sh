#!/bin/bash

# Define the parameters
#cofs=(0.0 0.4 1.0 10.0)
#aspectRatio=(1.0 1.5 2.0 2.5 3.0)
phis=(0.5 0.6 0.7 0.8 0.9)
aspectRatio=(1.0 1.5 2.0 2.5 3.0)
cofs=(0.4)

# Define the maximum number of parallel tasks
max_parallel_tasks=20

echo "Start of the loop"

# Loop through the values of -var flag for COF
for cof in "${cofs[@]}"
do
    # Loop through the values of -var flag for ap
    for ap in "${aspectRatio[@]}"
    do
        # Loop through the values of -var flag for phi
        for phi in "${phis[@]}"
        do
            # Submit the job directly to sbatch
            sbatch <<EOL
#!/bin/bash
#SBATCH -n 1 #Request 1 task (core)
#SBATCH -t 0-01:00 #Request runtime of 1 hour
##SBATCH -o output_post_%j.txt #redirect output to output_post_JOBID.txt
#SBATCH -e error_post_%j.txt #redirect errors to error_post_JOBID.txt
python main.py -c $cof -a $ap -t phi -v $phi
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
