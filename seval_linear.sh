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

nohup /home/lazarosg/miniconda3/envs/ijepa/bin/python peval_linear.py 
  --fname cls_configs/clsin100.yaml 
  &> logs/in100/oin100-peval_linear.out &


nohup /media/data/lazarosg/miniconda3/envs/ijepa/bin/python peval_linear.py 
  --fname cls_configs/clsiic.yaml &> logs_PKT/iic-train-L2/oclsiic-pfeature_extractor_try.out

# feature extraction for malena2
nohup /media/data/lazarosg/miniconda3/envs/ijepa/bin/python pfeature_extractor.py  
   --fname cls_configs/clsiic.yaml 
  &> logs_PKT_chunks/iic-train-L2_PKT_chunks-seed21/ocls-iic-L2-PKT-chunks-sout.out &

nohup /media/data/lazarosg/miniconda3/envs/ijepa/bin/python pfeature_extractor.py  
   --fname cls_configs/clsiic.yaml &