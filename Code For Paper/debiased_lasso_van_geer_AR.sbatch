#!/bin/bash
#SBATCH --job-name dl_van_AR        # Set a name for your job. This is especially useful if you have multiple jobs queued.
#SBATCH --partition medium         # Slurm partition to use
#SBATCH --ntasks 10              # Number of tasks to run. By default, one CPU core will be allocated per task
#SBATCH --time 13-11:50             # Wall time limit in D-HH:MM
#SBATCH --mem-per-cpu=1000           # Memory limit for each tasks (in MB)
#SBATCH -o out/dl_van_AR_%j.out        # File to which STDOUT will be written
#SBATCH -e out/dl_van_AR_%j.err        # File to which STDERR will be written
#SBATCH --mail-type=ALL            # Type of email notification- NONE,BEGIN,END,FAIL,ALL
#SBATCH --mail-user=yikun@uw.edu # Email to which notifications will be sent

module load R
i=${SLURM_ARRAY_TASK_ID}
Rscript ./debiased_lasso_van_geer_AR.R ${i}
