#!/bin/bash

#SBATCH -J IJEPALPSTL10
#SBATCH -t 6:00:00
#SBATCH --mem=256G
#SBATCH -c 16
#SBATCH -n 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lazarosg@csd.auth.gr
#SBATCH -p ampere
#SBATCH --gres=gpu:1


module load gcc miniconda3 cuda
source $CONDA_PROFILE/conda.sh
conda activate ijepa
export PATH=$CONDA_PREFIX/bin:$PATH

python pmulti-linear-probing-STL10.py \
  --fname cls_configs/cls-in100-stl10-multi.yaml