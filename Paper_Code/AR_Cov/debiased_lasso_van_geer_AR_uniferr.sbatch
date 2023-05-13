#!/bin/bash
#SBATCH --job-name dlvan_AR_unif        # Set a name for your job. This is especially useful if you have multiple jobs queued.
#SBATCH --partition medium         # Slurm partition to use
#SBATCH --array=1-500%200               # How many tasks in the array
#SBATCH --time 11-11:50             # Wall time limit in D-HH:MM
#SBATCH --mem-per-cpu=1000           # Memory limit for each tasks (in MB)
#SBATCH -o out/dlvan_AR_unif_%j.out        # File to which STDOUT will be written
#SBATCH -e out/dlvan_AR_unif_%j.err        # File to which STDERR will be written
#SBATCH --mail-type=ALL            # Type of email notification- NONE,BEGIN,END,FAIL,ALL
#SBATCH --mail-user=yikun@uw.edu  # Email to which notifications will be sent

module load R
i=${SLURM_ARRAY_TASK_ID}
Rscript ./debiased_lasso_van_geer_AR_uniferr.R ${i}
