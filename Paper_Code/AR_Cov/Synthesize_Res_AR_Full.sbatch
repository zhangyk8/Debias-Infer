#!/bin/bash
#SBATCH --exclude cls-cmp-a14,cls-cmp-a13
#SBATCH --job-name syn_res_AR_full        # Set a name for your job. This is especially useful if you have multiple jobs queued.
#SBATCH --partition short         # Slurm partition to use
#SBATCH --ntasks 3             # Number of tasks to run. By default, one CPU core will be allocated per task
#SBATCH --time 0-08:55             # Wall time limit in D-HH:MM
#SBATCH --mem-per-cpu=1000           # Memory limit for each tasks (in MB)
#SBATCH -o out/syn_res_AR_full.out        # File to which STDOUT will be written
#SBATCH -e out/syn_res_AR_full.err        # File to which STDERR will be written
#SBATCH --mail-type=ALL            # Type of email notification- NONE,BEGIN,END,FAIL,ALL
#SBATCH --mail-user=yikun@uw.edu   # Email to which notifications will be sent

source /etc/profile.d/z00_lmod.sh
module load Python
source ../yikun_venv/bin/activate
python ./Synthesize_Res_AR_Full.py
