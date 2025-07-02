sleep 30m

nohup /media/data/lazarosg/miniconda3/envs/ijepa/bin/python main.py     --fname configs/iic-train.yaml     --devices cuda:0 cuda:1 &> logs_PKT_chunks/iic-train-PKT-chunks-scale10e0-seed43/oiic-PRETRAIN.sout &