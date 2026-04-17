#!/bin/bash
#SBATCH --job-name Hou2023_SAS_AR       # Set a name for your job. This is especially useful if you have multiple jobs queued.
#SBATCH --partition short         # Slurm partition to use
#SBATCH --array=1-1000               # How many tasks in the array
#SBATCH --time 0-11:10             # Wall time limit in D-HH:MM
#SBATCH --mem-per-cpu=6000           # Memory limit for each tasks (in MB)
#SBATCH -o out/Hou2023_SAS_AR_%j.out        # File to which STDOUT will be written
#SBATCH -e out/Hou2023_SAS_AR_%j.err        # File to which STDERR will be written
#SBATCH --mail-type=ALL            # Type of email notification- NONE,BEGIN,END,FAIL,ALL
#SBATCH --mail-user=yikun@uw.edu # Email to which notifications will be sent

module load Python
source ../yikun_venv/bin/activate
i=${SLURM_ARRAY_TASK_ID}
python ./Hou2023_SAS_Inf_AR.py ${i}

