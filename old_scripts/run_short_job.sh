#!/bin/bash

#SBATCH -J IJEPA_short_job
#SBATCH -p ampere
#SBATCH -t 06:00:00
#SBATCH --mem=48G
#SBATCH --mail-type=ALL
#SBATCH --mail-user lazarosg@csd.auth.gr
#SBATCH -n 2
#SBATCH --ntasks-per-node 4
#SBATCH -c 32

python main_distributed.py \
	--fname configs/tin_vith16_ep5.yaml \
	--folder logs/tin_A100/ \
	--partition ampere \
	--nodes 2 \
	--tasks-per-node 4 \
	--time 06:00:00

# tin_vith16.64-bs.128-ep.5