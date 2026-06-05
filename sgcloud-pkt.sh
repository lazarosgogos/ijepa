#!/bin/bash

#SBATCH -J in1k-IJEPAL2PKT
#SBATCH -t 5-00:00:00
#SBATCH --mem=128G
#SBATCH -c 10
#SBATCH -n 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lazarosg@csd.auth.gr
#SBATCH -p h100
#SBATCH --gpus=2


module load gcc/14.2.0 python/3.14.0 cuda/12.8.1
source myenv/bin/activate

python main.py  \
	--fname configs/in1k_vitb16_ep300-pkt.yaml \
	--devices cuda:0 cuda:1 # cuda:2 cuda:3
