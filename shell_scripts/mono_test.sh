#!/bin/bash

#SBATCH -J IJEPATEST
#SBATCH -t 50:00
#SBATCH --mem=128G
#SBATCH -c 16
#SBATCH -n 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lazarosg@csd.auth.gr
#SBATCH -p ampere
#SBATCH --gres=gpu:4

module load gcc miniconda3 cuda
source $CONDA_PROFILE/conda.sh
conda activate ijepa
export PATH=$CONDA_PREFIX/bin:$PATH

python main.py  \
	--fname configs/in100_vitb16_ep600.yaml \
	--devices cuda:0 cuda:1 cuda:2 cuda:3
