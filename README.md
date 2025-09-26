# Simple Shear Project

This repository contains the code and configuration for running simulations of simple shear flow of spheroids, as part of a research project supporting the publication "".
 The dataset generated from these simulations is available on [Zenodo](https://zenodo.org/records/17140603).

## Project Overview

This project is designed to run simulations using the LIGGGHTS software for discrete element method (DEM) simulations of simple shear flow. It includes scripts for launching simulations, compiling the LIGGGHTS source code, and post-processing results. The project is configured to run on a high-performance computing (HPC) cluster using SLURM, but it can also be adapted for local execution.

## Repository Structure

- **`launcher/`**: Contains scripts for launching simulations on the HPC cluster using SLURM.
- **`liggghts_source/`**: Contains the LIGGGHTS source code and a `compiler.sh` script to automatically configure and compile LIGGGHTS using CMake with the correct flags.
- **`postprocessing/`**: Contains post-processing scripts and a `requirements.txt` file for Python dependencies. A script to automatically generate a Python virtual environment (`.venv`) and install dependencies will be added.
- **`config.yaml`**: Configuration file specifying simulation parameters, sweep parameters, and SLURM settings.
- **`.gitignore`**: Specifies files and directories to be ignored by Git.
- **`modules_used.txt`**: Lists required HPC modules and their versions for compiling and running LIGGGHTS.
- **`setup_env.sh`**: Script to automatically load required HPC modules.

## Prerequisites

### On an HPC Cluster
The project is configured to run on an HPC cluster with the modules (listed in `modules_used.txt`).
They can be automatically loaded by running 
```bash
source setup_env.sh
```

### On a Local Machine
If running locally, you need to install the above dependencies manually. Refer to the official documentation for each dependency to install compatible versions:

LIGGGHTS
CMake
Eigen
VTK
Python 3.11
Other dependencies (e.g., gcc, zlib-ng, bzip2, pkgconf)

## Setup Instructions

### 1. Configure the Simulation
Edit the `config.yaml` file to set the desired parameters:

- **Simulation Data Directory**: Update the `sim_data_dir` field to specify where simulation data should be stored. For example:
  ```yaml
  sim_data_dir: "/path/to/your/storage/sim_data"
  ```
- **Email Notifications**: Update the mail_user field in the SLURM settings to receive job status notifications (e.g., BEGIN, END, FAIL):
```yaml
mail_user: "your.email@example.com"
```

- **Other Parameters**: The config.yaml file also includes:
    - executable: Path to the LIGGGHTS executable (/Liggghts_dev/build/liggghts).
    - input_script: LIGGGHTS input script (in.simple_shear_le).
    - simulation parameters: radius (1), density (1000).
    - wweep parameters: aspect_ratios, inertial_numbers, cofs (coefficients of friction).
    - SLURM settings: ntasks, cpus_per_task, mem, time, mail_type.


### 2. Compile LIGGGHTS
Navigate to the `liggghts_source/` directory and run the `compiler.sh` script to compile LIGGGHTS:
```bash
cd liggghts_source
./compiler.sh
```
This script uses CMake with the flags needed to launch the simulations and builds the LIGGGHTS executable.

### 3. Run Simulations
1. Ensure the required HPC modules are loaded:
```bash
source ../setup_env.sh
```

2. Navigate to the launcher/ directory and submit the SLURM job:
```bash
cd launcher
sbatch <submission_script>
```
Replace <submission_script> with the appropriate SLURM script in the launcher/ directory.

### 4. Post-Processing

Navigate to the `postprocessing/` directory and activate the virtual environment:
```bash
cd postprocessing
source setup_venv.sh
```

Run post-processing scripts with:
```bash
bash parallel_postprocessing_sbatch.sh
```
The data will appear in the `output_data_hertz/` directory.


### License
See the LICENSE file for details.

### Contact
For questions or issues, please contact [me](jacopobil8@gmail.com).