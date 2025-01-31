#!/bin/bash

#SBATCH -J ijepabirds
#SBATCH -t 6-00:00:00
#SBATCH --mem=64G
#SBATCH -c 16
#SBATCH -n 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lazarosg@csd.auth.gr
#SBATCH -p ampere
#SBATCH --gres=gpu:4
#SBATCH --qos=ampere-extd

module load gcc miniconda3 cuda
source $CONDA_PROFILE/conda.sh
conda activate ijepa
export PATH=$CONDA_PREFIX/bin:$PATH

python main.py  \
	--fname configs/birds_A100.yaml \
	--devices cuda:0 cuda:1 cuda:2 cuda:3
