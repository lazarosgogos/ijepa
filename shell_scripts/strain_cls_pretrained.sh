nohup /media/data/lazarosg/miniconda3/envs/ijepa/bin/python pcls_pretrained_vitb.py 
    &> logs/iic-train-PKT/ocls_pretrained_vitb_PKT.out &

nohup /media/data/lazarosg/miniconda3/envs/ijepa/bin/python pcls_pretrained_vitb_CIFAR10.py 
    &> logs/iic-train-double/ocls_pretrained_vitb_CIFAR10.out& 

nohup /media/data/lazarosg/miniconda3/envs/ijepa/bin/python pcls_pretrained_vitb.py 
    &> logs/iic-train-double/ocls_pretrained_vitb_first.out &

nohup /media/data/lazarosg/miniconda3/envs/ijepa/bin/python pcls_pretrained_vitb_CIFAR10.py 
    &> logs/iic-train-PKT/ocls_pretrained_vitb_withPKT_onIIC_CIFAR10.out& 

# train on COSINE sim loss to see what happened
nohup /media/data/lazarosg/miniconda3/envs/ijepa/bin/python pcls_pretrained_vitb.py 
    &> logs/iic-train-cosine/ocls_pretrained_vitb_COSINE.out & 

nohup /media/data/lazarosg/miniconda3/envs/ijepa/bin/python pcls_pretrained_vitb.py 
    &> logs/iic-train-PKT/ocls_pretrained_vitb-L2-PKT-scaled.out &

nohup /media/data/lazarosg/miniconda3/envs/ijepa/bin/python pcls_pretrained_vitb.py 
    &> cls_logs/ocls_pretrained_vitb_from100to200.out &

nohup /home/lazarosg/miniconda3/envs/ijepa/bin/python pfeature_extractor.py 
  --fname cls_configs/clsiic.yaml &> logs_PKT/iic-train-L2/oclsiic-pfeature_extractor_try.out &