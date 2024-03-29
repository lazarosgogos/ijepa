#!/bin/bash

#SBATCH -J IJEPA_one_GPU
#SBATCH -p ampere
#SBATCH -t 05:00
#SBATCH --mem=16G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lazarosg@csd.auth.gr
#SBATCH -n 1
#SBATCH --ntasks-per-node 4
#SBATCH -c 4

python main_distributed.py \
	--fname configs/tin_vith16_ep5.yaml \
	--folder logs/tin_vith16.64-bs.128-ep.5/ \
	--partition ampere \
	--nodes 1 \
	--tasks-per-node 1 \
	--time 05:00
