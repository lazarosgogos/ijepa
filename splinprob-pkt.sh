#!/bin/bash

#SBATCH -J IJEPA_IN100_L2_LIN_PROB_PKT
#SBATCH -t 6:00:00
#SBATCH --mem=256G
#SBATCH -c 16
#SBATCH -n 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lazarosg@csd.auth.gr
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --dependency=afterok:1855986

module load gcc miniconda3 cuda
source $CONDA_PROFILE/conda.sh
conda activate ijepa
export PATH=$CONDA_PREFIX/bin:$PATH

python pmulti-linear-probing.py \
  --fname cls_configs/cls-in100-multi-pkt.yaml