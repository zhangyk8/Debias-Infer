#!/bin/bash
#SBATCH --exclude cls-cmp-a14,cls-cmp-a13
#SBATCH --job-name la_ref_Equi       # Set a name for your job. This is especially useful if you have multiple jobs queued.
#SBATCH --partition medium         # Slurm partition to use
#SBATCH --array=1-500               # How many tasks in the array
#SBATCH --time 4-11:10             # Wall time limit in D-HH:MM
#SBATCH --mem-per-cpu=1000           # Memory limit for each tasks (in MB)
#SBATCH -o out/la_ref_Equi_%j.out        # File to which STDOUT will be written
#SBATCH -e out/la_ref_Equi_%j.err        # File to which STDERR will be written
#SBATCH --mail-type=ALL            # Type of email notification- NONE,BEGIN,END,FAIL,ALL
#SBATCH --mail-user=yikun@uw.edu # Email to which notifications will be sent

source /etc/profile.d/z00_lmod.sh
module load Python
source ../yikun_venv/bin/activate
i=${SLURM_ARRAY_TASK_ID}
python ./Lasso_refit_Equi.py ${i}

