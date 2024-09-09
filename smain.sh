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
    &> logs/iic-train-PKT/oiic-train-L2-PKTscaled.out &

nohup /media/data/lazarosg/miniconda3/envs/ijepa/bin/python main.py 
    --fname configs/iic-train.yaml 
    --devices cuda:0 cuda:1 
    &> logs_PKT/iic-train-PKT/oiic-train-PKT.out &

# from cidl19
nohup /home/lazarosg/miniconda3/envs/ijepa/bin/python main.py --fname configs/in.yaml --devices cuda:0 &> logs/in100/oin100-vits.out &
