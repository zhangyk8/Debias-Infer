#!/bin/bash
#SBATCH --job-name rj_bul_cir_uniferr        # Set a name for your job. This is especially useful if you have multiple jobs queued.
#SBATCH --partition short         # Slurm partition to use
#SBATCH --array=1-1000               # How many tasks in the array
#SBATCH --time 0-11:50             # Wall time limit in D-HH:MM
#SBATCH --mem-per-cpu=900           # Memory limit for each tasks (in MB)
#SBATCH -o out/rj_bul_cir_uniferr_%j.out        # File to which STDOUT will be written
#SBATCH -e out/rj_bul_cir_uniferr_%j.err        # File to which STDERR will be written
#SBATCH --mail-type=ALL            # Type of email notification- NONE,BEGIN,END,FAIL,ALL
#SBATCH --mail-user=yikun@uw.edu # Email to which notifications will be sent

module load R
i=${SLURM_ARRAY_TASK_ID}
Rscript ./ridge_proj_bulhmann_cirsym_uniferr.R ${i}
