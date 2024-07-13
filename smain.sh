nohup /media/data/lazarosg/miniconda3/envs/ijepa/bin/python main.py 
    --fname configs/iic-train.yaml 
    --devices cuda:0 cuda:1 
    &>> logs/iic-train-double/oiic-train-double.out &


# train with tiny ViT
nohup /media/data/lazarosg/miniconda3/envs/ijepa/bin/python main.py 
    --fname configs/iic-train-tiny.yaml 
    --devices cuda:0 cuda:1 
    &> logs/iic-train-tiny/oiic-train-tiny.out &

# train with small ViT
nohup /media/data/lazarosg/miniconda3/envs/ijepa/bin/python main.py 
    --fname configs/iic-train-small.yaml 
    --devices cuda:0 cuda:1 
    &> logs/iic-train-small/oiic-train-small.out &

nohup /media/data/lazarosg/miniconda3/envs/ijepa/bin/python main.py 
    --fname configs/iic-train.yaml 
    --devices cuda:0 cuda:1 
    &> logs/iic-train-cosine/oiic-train-cosine.out &

nohup /media/data/lazarosg/miniconda3/envs/ijepa/bin/python main.py 
    --fname configs/iic-train.yaml 
    --devices cuda:0 cuda:1 
    &> logs/iic-train-PKT/oiic-train-L2_PKT.out &