#!/bin/bash
#SBATCH --job-name la_rf_AR_gauss        # Set a name for your job. This is especially useful if you have multiple jobs queued.
#SBATCH --partition short         # Slurm partition to use
#SBATCH --array=1-1000               # How many tasks in the array
#SBATCH --time 0-11:10             # Wall time limit in D-HH:MM
#SBATCH --mem-per-cpu=1000           # Memory limit for each tasks (in MB)
#SBATCH -o out/la_rf_AR_gauss_%j.out        # File to which STDOUT will be written
#SBATCH -e out/la_rf_AR_gauss_%j.err        # File to which STDERR will be written
#SBATCH --mail-type=ALL            # Type of email notification- NONE,BEGIN,END,FAIL,ALL
#SBATCH --mail-user=yikun@uw.edu # Email to which notifications will be sent

module load Python
source ../yikun_venv/bin/activate
i=${SLURM_ARRAY_TASK_ID}
python ./Lasso_refit_AR.py ${i}

