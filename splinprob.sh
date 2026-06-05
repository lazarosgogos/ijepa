#!/bin/bash

#SBATCH -J IJEPALP
#SBATCH -t 1-00:00:00
#SBATCH --mem=256G
#SBATCH -c 10
#SBATCH -n 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lazarosg@csd.auth.gr
#SBATCH --partition=a100
#SBATCH --gpus=1
#SBATCH --dependency=afterok:17584

# module load gcc/13.2.0  miniconda3 cuda
# source $CONDA_PROFILE/conda.sh
# conda activate ijepa
# export PATH=$CONDA_PREFIX/bin:$PATH

module load gcc/14.2.0 python/3.14.0 cuda/12.8.1
source myenv/bin/activate

python pmulti-linear-probing.py \
  --fname cls_configs/cls-in1k-multi.yaml
