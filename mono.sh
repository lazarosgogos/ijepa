#!/bin/bash

#SBATCH -J IJEPA_short_job
#SBATCH -t 05:59:00
#SBATCH --mem=256G
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

python main.py  \
	--fname configs/tin_vith16_ep5.yaml \
	--devices cuda:0
