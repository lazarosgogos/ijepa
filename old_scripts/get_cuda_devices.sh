#!/bin/bash

#SBATCH -J CUDA_avail
#SBATCH -t 01:00
#SBATCH --mem=2G
#SBATCH -c 1
#SBATCH -n 1
#SBATCH --mail-type=NONE
#SBATCH --mail-user=lazarosg@csd.auth.gr
#SBATCH -p ampere
#SBATCH --gres=gpu:2

module load gcc miniconda3 cuda
source $CONDA_PROFILE/conda.sh
conda activate ijepa
export PATH=$CONDA_PREFIX/bin:$PATH

python print_cuda_devs.py
