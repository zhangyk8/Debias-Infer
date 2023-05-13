#!/bin/bash
#SBATCH --job-name dljav_AR_gauss        # Set a name for your job. This is especially useful if you have multiple jobs queued.
#SBATCH --partition medium         # Slurm partition to use
#SBATCH --array=1-1000               # How many tasks in the array
#SBATCH --time 3-11:40             # Wall time limit in D-HH:MM
#SBATCH --mem-per-cpu=1000           # Memory limit for each tasks (in MB)
#SBATCH -o out/dljav_AR_gauss_%j.out        # File to which STDOUT will be written
#SBATCH -e out/dljav_AR_gauss_%j.err        # File to which STDERR will be written
#SBATCH --mail-type=ALL            # Type of email notification- NONE,BEGIN,END,FAIL,ALL
#SBATCH --mail-user=yikun@uw.edu # Email to which notifications will be sent

module load R
i=${SLURM_ARRAY_TASK_ID}
Rscript ./debiased_lasso_javanmard_AR.R ${i}
