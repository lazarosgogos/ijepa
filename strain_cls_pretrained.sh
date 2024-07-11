nohup /media/data/lazarosg/miniconda3/envs/ijepa/bin/python pcls_pretrained_vitb.py 
    &> logs/iic-train-PKT/ocls_pretrained_vitb_PKT.out &

nohup /media/data/lazarosg/miniconda3/envs/ijepa/bin/python pcls_pretrained_vitb_CIFAR10.py 
    &> logs/iic-train-double/ocls_pretrained_vitb_CIFAR10.out& 

nohup /media/data/lazarosg/miniconda3/envs/ijepa/bin/python pcls_pretrained_vitb.py 
    &> logs/iic-train-double/ocls_pretrained_vitb_first.out &