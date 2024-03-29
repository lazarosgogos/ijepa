#!/bin/bash

#SBATCH -J IJEPA_pretrained
#SBATCH -t 15:00
#SBATCH --mem=512G
#SBATCH -c 32
#SBATCH -n 1
#SBATCH --mail-type=NONE
#SBATCH --mail-user=lazarosg@csd.auth.gr
#SBATCH -p ampere
#SBATCH --gres=gpu:1

module load gcc miniconda3 cuda
source $CONDA_PROFILE/conda.sh
conda activate ijepa
export PATH=$CONDA_PREFIX/bin:$PATH

python pretrained_vith.py
