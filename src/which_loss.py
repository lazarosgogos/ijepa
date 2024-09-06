import torch
import torch.nn.functional as F

def L2(z,h):
  """ Calculate the L2 loss between z and h, then return it.
  :param z: the representation of the patches after being passed through 
   the Context Encoder and the Predictor
  :param h: the representation of the patches after being passed through
   the Target Encoder """
  loss_l2 = F.smooth_l1_loss(z,h)
  return loss_l2

def PKT(z,h):
  """ Calculate the PKT loss. WIP
   
  :param z: the representation of the patches after being passed through 
   the Context Encoder and the Predictor
  :param h: the representation of the patches after being passed through
   the Target Encoder """
  pass