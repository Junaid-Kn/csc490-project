#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --gres=gpu:4         # Number of GPU(s) per node
#SBATCH --cpus-per-task=16         # CPU cores/threads
#SBATCH --mem=20G               # memory per node
#SBATCH --time=0-12:00:00         # time (DD-HH:MM)
#SBATCH --output=kd_model.out    # output file name (%j expands to jobId)

module load python cuda scipy-stack
source ./env/bin/activate


python -m ./csc490-project/kd_src/csc490.py