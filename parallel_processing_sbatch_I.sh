#!/bin/bash

# Define the parameters
aspectRatio=(1.0 1.5 2.0 2.5 3.0)
cofs=(0 0.4 1.0 10.0)
#Is=(0.1 0.0398 0.0158 0.0063 0.0025 0.001)
Is=(0.0063 0.0025 0.001)


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
#SBATCH -t 0-01:00 #Request runtime of 1 hour
##SBATCH -o output_post_%j.txt #redirect output to output_post_JOBID.txt
#SBATCH -e error_post_%j.txt #redirect errors to error_post_JOBID.txt
python main.py -c $cof -a $ap -t I -v $I
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
