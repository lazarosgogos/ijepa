#!/bin/bash

#SBATCH -J vith_load_test
#SBATCH -t 5:00
#SBATCH --mem=32G
#SBATCH -c 16
#SBATCH -n 1
#SBATCH --mail-type=END
#SBATCH --mail-user=lazarosg@csd.auth.gr
#SBATCH -p gpu
#SBATCH --gres=gpu:1

module load gcc miniconda3 cuda
source $CONDA_PROFILE/conda.sh
conda activate ijepa
export PATH=$CONDA_PREFIX/bin:$PATH

python pretrained_vith.py
