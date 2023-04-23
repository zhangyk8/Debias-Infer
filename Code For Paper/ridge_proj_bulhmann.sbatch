#!/bin/bash
#SBATCH --job-name rj_bulh        # Set a name for your job. This is especially useful if you have multiple jobs queued.
#SBATCH --partition medium         # Slurm partition to use
#SBATCH --ntasks 21              # Number of tasks to run. By default, one CPU core will be allocated per task
#SBATCH --time 6-11:50             # Wall time limit in D-HH:MM
#SBATCH --mem-per-cpu=1000           # Memory limit for each tasks (in MB)
#SBATCH -o out/rj_bulh.out        # File to which STDOUT will be written
#SBATCH -e out/rj_bulh.err        # File to which STDERR will be written
#SBATCH --mail-type=ALL            # Type of email notification- NONE,BEGIN,END,FAIL,ALL
#SBATCH --mail-user=yikun@uw.edu # Email to which notifications will be sent

module load R
Rscript ./ridge_proj_bulhmann.R
