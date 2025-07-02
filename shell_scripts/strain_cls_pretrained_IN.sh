nohup /media/data/lazarosg/miniconda3/envs/ijepa/bin/python pcls_pretrained_vith_IN.py &> cls_pretrained_vith_IN.out &

nohup /media/data/lazarosg/miniconda3/envs/ijepa/bin/python pcls_pretrained_vith_IN22k.py &> cls_pretrained__vith_IN22k.out &

#SBATCH --dependency=afterok:1932831