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
    --devices cuda:0 cuda:1 &
    &> logs_PKT/iic-train-PKT-seed-15/oiic-train-PKT-seed-15.out &

nohup /media/data/lazarosg/miniconda3/envs/ijepa/bin/python main.py 
    --fname configs/iic-train.yaml 
    --devices cuda:0 cuda:1 
    &> logs_PKT_chunks/iic-train-L2_PKT_chunks-seed43/oiic-pretrain-L2_PKT_chunks-sout.out &


# from cidl19
nohup /home/lazarosg/miniconda3/envs/ijepa/bin/python main.py --fname configs/in.yaml --devices cuda:0 &> logs/in100/oin100-vits.out &

nohup /home/lazarosg/miniconda3/envs/ijepa/bin/python main.py --fname configs/stl-train.yaml --devices cuda:0 &> logs_STL_ep500/stl-l2-seed0/ostl-PRETRAIN.out &

# evaulation
nohup /media/data/lazarosg/miniconda3/envs/ijepa/bin/python main.py     --fname configs/iic-eval.yaml     --devices cuda:0 cuda:1     --eval 1     &> logs_PKT_full/iic-train-PKT-seed21/evaluation.out &