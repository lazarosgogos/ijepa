import torch
import torch.nn.functional as F

from src import PKT as PKTClass

def L2(z,h, num_pred_masks=4):
  """ Calculate the L2 loss between z and h, then return it.
  :param z: the representation of the patches after being passed through 
   the Context Encoder and the Predictor
  :param h: the representation of the patches after being passed through
   the Target Encoder """
  loss_l2 = F.smooth_l1_loss(z,h)
  return loss_l2

def PKT(z,h, num_pred_masks=4):
  """ Calculate the PKT loss. WIP
   
  :param z: the representation of the patches after being passed through 
   the Context Encoder and the Predictor
  :param h: the representation of the patches after being passed through
   the Target Encoder """
  # loss_PKT = PKT.cosine_similarity_loss()

  # PKT first shot
  loss_pkt = 0
  first_dim = z.size(0)
  z = z.view(first_dim//num_pred_masks, num_pred_masks, *z.size()[1:])
  h = h.view(first_dim//num_pred_masks, num_pred_masks, *h.size()[1:])
  # suddenly, now instead of [256, 20, 768] we have [64, 4, 20, 768]
  for i in range(z.size(0)):
      z_ = z[i] # get element i which would be [4, 20, 768] in size
      h_ = h[i]
      emb_size = z.size(-1)
      # t = t.view(big_b_s//num_patches, num_patches, *t.size()[1:])
      z_ = z_.view(-1, emb_size) # flatten it, without changing anything
      h_ = h_.view(-1, emb_size)
      loss_pkt += PKTClass.cosine_similarity_loss(z_,h_)
  # loss_pkt /= z.size(0) # normalize by batch size

  # loss_pkt *= 100 # scale PKT to match l2 loss and equalize the effect

  # -- Other thoughts to try out
  # (64*4, 20, EMB_SIZE: 768) # z -> [64*4*20, 768] or [64*4, 768]
  # OR [64,4,768] -> mean [64, 768]
  # (64*4, 20, EMB_SIZE: 768) # h
  # .view() 
  # alpha = .1 * cosine_similarity_loss
  # loss = AllReduce.apply(loss_l2 + loss_pkt) 
  return loss_pkt

  """
  for i in range(batch_size):
      PKT([4, 20, 768] [4, 20, 768])
      [80, 768] @ [768, 80] = [80,80] # diagonal = 1

  # (64*4, 20, EMB_SIZE: 768) # z -> [64*4*20, 768] or [64*4, 768]

  """

def L2_PKT(z,h, num_pred_masks=4):
  """ Calculate the PKT loss. WIP
   
  :param z: the representation of the patches after being passed through 
   the Context Encoder and the Predictor
  :param h: the representation of the patches after being passed through
   the Target Encoder """

  loss_l2 = L2(z,h)
  loss_pkt = PKT(z,h)
  return loss_l2 + loss_pkt

def L2_PKT_scaled(z,h, num_pred_masks=4, **kwargs):
  """ Calculate the PKT loss. WIP
   
  :param z: the representation of the patches after being passed through 
   the Context Encoder and the Predictor
  :param h: the representation of the patches after being passed through
   the Target Encoder """
  pkt_scale = float(kwargs.get('pkt_scale', 1.)) # default to 1 if not present
  loss_l2 = L2(z,h)
  loss_pkt = PKT(z,h)
  return loss_l2 + loss_pkt * pkt_scale


def PKT_full(z,h, num_pred_masks=4):
  """ Calculate the PKT loss between all patches of all batches, at once.
   
  :param z: the representation of the patches after being passed through 
   the Context Encoder and the Predictor
  :param h: the representation of the patches after being passed through
   the Target Encoder """
  
  # we have [256, 20, 768]
  # we can view it as [256*20, 768] = [5120, 768]
  # and then perform PKT there
  # or take mean of second dimension and perform PKT over [256, 768]
  emb_size = z.size(-1)
  z_ = z.view(-1, emb_size)
  h_ = h.view(-1, emb_size)
  loss_pkt_full = PKTClass.cosine_similarity_loss(z_, h_)
  return loss_pkt_full
"""
  # PKT between all patches of whole batch
  loss_pkt = 0
  first_dim = z.size(0)
  z = z.view(first_dim//num_pred_masks, num_pred_masks, *z.size()[1:])
  h = h.view(first_dim//num_pred_masks, num_pred_masks, *h.size()[1:])
  # suddenly, now instead of [256, 20, 768] we have [64, 4, 20, 768]
  for i in range(z.size(0)):
      z_ = z[i] # get element i which would be [4, 20, 768] in size
      h_ = h[i]
      emb_size = z.size(-1)
      # t = t.view(big_b_s//num_patches, num_patches, *t.size()[1:])
      z_ = z_.view(-1, emb_size) # flatten it, without changing anything
      h_ = h_.view(-1, emb_size)
      loss_pkt += PKTClass.cosine_similarity_loss(z_,h_)
  loss_pkt /= z.size(0) # normalize by batch size
  # loss_pkt *= 100 # scale PKT to match l2 loss and equalize the effect

  # -- Other thoughts to try out
  # (64*4, 20, EMB_SIZE: 768) # z -> [64*4*20, 768] or [64*4, 768]
  # OR [64,4,768] -> mean [64, 768]
  # (64*4, 20, EMB_SIZE: 768) # h
  # .view() 
  # alpha = .1 * cosine_similarity_loss
  # loss = AllReduce.apply(loss_l2 + loss_pkt) 
  return loss_pkt
  """

"""
[80, 768] @ [768, 80] = [80, 80] * 64 / 64

[5120, 768] @ [768, 5120] =  [5120, 5120] """