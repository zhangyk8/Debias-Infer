#!/bin/bash
#SBATCH --exclude cls-cmp-a14,cls-cmp-a13
#SBATCH --job-name deb_cir_Rver        # Set a name for your job. This is especially useful if you have multiple jobs queued.
#SBATCH --partition medium         # Slurm partition to use
#SBATCH --array=1-1000               # How many tasks in the array
#SBATCH --time 10-11:10             # Wall time limit in D-HH:MM
#SBATCH --mem-per-cpu=1200           # Memory limit for each tasks (in MB)
#SBATCH -o out/deb_cir_Rver%j.out        # File to which STDOUT will be written
#SBATCH -e out/deb_cir_Rver%j.err        # File to which STDERR will be written
#SBATCH --mail-type=ALL            # Type of email notification- NONE,BEGIN,END,FAIL,ALL
#SBATCH --mail-user=yikun@uw.edu # Email to which notifications will be sent

source /etc/profile.d/z00_lmod.sh
module load R
i=${SLURM_ARRAY_TASK_ID}
python ./Debiased_Inf_CirSym_Rversion.R ${i}

