from src.models.vision_transformer import vit_huge
from src import helper
# Initialize the ViT-H model with the specified patch size and resolution
model = vit_huge(patch_size=14, num_classes=1000) # Adjust num_classes if needed

"""def init_model(
    device,
    patch_size=16,
    model_name='vit_base',
    crop_size=224,
    pred_depth=6,
    pred_emb_dim=384
):"""

encoder, predictor = helper.init_model(device='cuda', 
                                       patch_size=4,
                                       model_name='vit_huge',
                                       crop_size=64,
                                       pred_depth=12,
                                       pred_emb_dim=384)



import torch
# Load the state dictionary from the file
load_path = '/content/IN1K-vit.h.14-300e.pth.tar'
ckpt = torch.load(load_path, map_location=torch.device('cpu'))
# state_dict = torch.load('/content/IN1K-vit.h.14-300e.pth.tar')
pretrained_dict = ckpt['encoder']
# -- loading encoder
for k, v in pretrained_dict.items():
  encoder.state_dict()[k[len('module.'):]].copy_(v) # encoder ???


# Load the state dictionary into the model
# model.load_state_dict(state_dict)

# Print the layers/modules of the model for inspection
def print_model_layers(model, prefix=''):
  for name, module in model.named_children():
    if isinstance(module, torch.nn.Module):
      module_name = prefix + '.' + name if prefix else name
      print(module_name)
      print_model_layers(module, prefix=module_name)

print_model_layers(model)