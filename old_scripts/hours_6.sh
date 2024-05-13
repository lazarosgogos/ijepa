#!/bin/bash

#SBATCH -J IJEPA_6_hours
#SBATCH -t 06:00:00
#SBATCH --mem=512G
#SBATCH -c 32
#SBATCH -n 1
#SBATCH --mail-type=END
#SBATCH --mail-user=lazarosg@csd.auth.gr
#SBATCH -p ampere
#SBATCH --gres=gpu:4

module load gcc miniconda3 cuda
source $CONDA_PROFILE/conda.sh
conda activate ijepa
export PATH=$CONDA_PREFIX/bin:$PATH

python main.py  \
	--fname configs/tin_vith16_ep5.yaml \
	--devices cuda:0 cuda:1 cuda:2 cuda:3
