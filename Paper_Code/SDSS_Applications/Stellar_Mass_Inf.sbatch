#!/bin/bash
#SBATCH --exclude cls-cmp-a14,cls-cmp-a13,cls-cmp-c16,cls-cmp-c8
#SBATCH --job-name sdss_app       # Set a name for your job. This is especially useful if you have multiple jobs queued.
#SBATCH --partition medium         # Slurm partition to use
#SBATCH --array=1-100               # How many tasks in the array
#SBATCH --time 11-11:40             # Wall time limit in D-HH:MM
#SBATCH --mem-per-cpu=1000           # Memory limit for each tasks (in MB)
#SBATCH -o out/sdss_app_%j.out        # File to which STDOUT will be written
#SBATCH -e out/sdss_app_%j.err        # File to which STDERR will be written
#SBATCH --mail-type=ALL            # Type of email notification- NONE,BEGIN,END,FAIL,ALL
#SBATCH --mail-user=yikun@uw.edu # Email to which notifications will be sent


source /etc/profile.d/z00_lmod.sh
module load Python
source ../yikun_venv/bin/activate
python ./Stellar_Mass_Inf.py
