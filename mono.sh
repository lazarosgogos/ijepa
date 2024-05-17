#!/bin/bash

#SBATCH -J IJEPA_100_EPOCHS
#SBATCH -t 6-00:00:00
#SBATCH --mem=64G
#SBATCH -c 16
#SBATCH -n 1
#SBATCH --mail-type=END
#SBATCH --mail-user=lazarosg@csd.auth.gr
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --qos=ampere-extd

module load gcc miniconda3 cuda
source $CONDA_PROFILE/conda.sh
conda activate ijepa
export PATH=$CONDA_PREFIX/bin:$PATH

python main.py  \
	--fname configs/tin_vith16_ep5.yaml \
	--devices cuda:0
