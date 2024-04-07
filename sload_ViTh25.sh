#!/bin/bash

#SBATCH -J DOWNSTREAM_TASK_IJEPA
#SBATCH -t 5-23:59:00
#SBATCH --mem=64G
#SBATCH -c 16
#SBATCH -n 1
#SBATCH --mail-type=END
#SBATCH --mail-user=lazarosg@csd.auth.gr
#SBATCH -p ampere
#SBATCH --qos=ampere-extd
#SBATCH --gres=gpu:2


module load gcc miniconda3 cuda
source $CONDA_PROFILE/conda.sh
conda activate ijepa
export PATH=$CONDA_PREFIX/bin:$PATH

python pretrained_vith.py
