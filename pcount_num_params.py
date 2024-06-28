import numpy as np
import torch
from src.models.vision_transformer import vit_huge
from src import helper
import pprint as pp

IMG_CROPSIZE = 224
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model_names = ['vit_tiny', 'vit_small', 'vit_base', 'vit_large', 'vit_huge', 'vit_giant']

res = dict()

# source: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

for name in model_names:
    encoder, predictor = helper.init_model( device=device, # if device=device doesn't work, try device='cuda:1' 
                                            patch_size=14,
                                            model_name=name,
                                            crop_size=IMG_CROPSIZE,
                                            pred_depth=12,
                                            pred_emb_dim=384)
    
    params = count_parameters(encoder)
    del encoder, predictor
    res[name] = params

pp.pp(res)