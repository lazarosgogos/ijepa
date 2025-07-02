#!/bin/bash

#SBATCH -J IJEPALPPKT
#SBATCH -t 6:00:00
#SBATCH --mem=256G
#SBATCH -c 16
#SBATCH -n 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lazarosg@csd.auth.gr
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --dependency=afterok:2286416

module load gcc/13.2.0  miniconda3 cuda
source $CONDA_PROFILE/conda.sh
conda activate ijepa
export PATH=$CONDA_PREFIX/bin:$PATH

python pmulti-linear-probing.py \
  --fname cls_configs/cls-in100-multi-pkt.yaml