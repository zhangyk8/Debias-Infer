#!/bin/bash
#SBATCH --job-name dl_van_equicor        # Set a name for your job. This is especially useful if you have multiple jobs queued.
#SBATCH --partition largemem         # Slurm partition to use
#SBATCH --array=1-1000%50               # How many tasks in the array
#SBATCH --time 3-11:10             # Wall time limit in D-HH:MM
#SBATCH --mem-per-cpu=1000           # Memory limit for each tasks (in MB)
#SBATCH -o out/dl_van_equicor_%j.out        # File to which STDOUT will be written
#SBATCH -e out/dl_van_equicor_%j.err        # File to which STDERR will be written
#SBATCH --mail-type=ALL            # Type of email notification- NONE,BEGIN,END,FAIL,ALL
#SBATCH --mail-user=yikun@uw.edu # Email to which notifications will be sent

module load R
i=${SLURM_ARRAY_TASK_ID}
Rscript ./debiased_lasso_van_geer_equicor.R ${i}
