#!/bin/bash

#SBATCH -J IJEPA_short_job
#SBATCH -t 05:00
#SBATCH --mem=512G
#SBATCH -c 64
#SBATCH -n 1
#SBATCH --mail-type=NONE
#SBATCH --mail-user=lazarosg@csd.auth.gr
#SBATCH -p ampere
#SBATCH --gres=gpu:4

module load gcc cuda/11.1.0
source $CONDA_PROFILE/conda.sh
conda activate ijepa
export PATH=$CONDA_PREFIX/bin:$PATH

python main_distributed.py \
	--fname configs/tin_vith16_ep5.yaml \
	--folder logs/tin_A100/ \
	--partition ampere \
	--nodes 1 \
	--tasks-per-node 1 \
	--time 1
