import torch
import torch.nn.functional as F

from src import PKT as PKTClass

import logging

logger = logging.getLogger()

def L2(z,h, **kwargs):
  """ Calculate the L2 loss between z and h, then return it.
  :param z: the representation of the patches after being passed through 
   the Context Encoder and the Predictor
  :param h: the representation of the patches after being passed through
   the Target Encoder """
  loss_l2 = F.smooth_l1_loss(z,h)
  return loss_l2

def PKT(z_init,h_init, **kwargs):
  """ Calculate the PKT loss. WIP
   
  :param z: the representation of the patches after being passed through 
   the Context Encoder and the Predictor
  :param h: the representation of the patches after being passed through
   the Target Encoder """
  # loss_PKT = PKT.cosine_similarity_loss()
  num_pred_masks = int(kwargs.get('num_pred_masks', 4))
  # PKT first shot
  variance_weight = kwargs.get('variance_weight', 0.) # use 0 by default
  loss_pkt = 0
  neg_variance_sum = 0
  first_dim = z_init.size(0)
  z = z_init.view(first_dim//num_pred_masks, num_pred_masks, *z_init.size()[1:])
  h = h_init.view(first_dim//num_pred_masks, num_pred_masks, *h_init.size()[1:])
  # suddenly, now instead of [256, 20, 768] we have [64, 4, 20, 768]
  for i in range(z.size(0)):
      z_ = z[i] # get element i which would be [4, 20, 768] in size
      h_ = h[i]
      emb_size = z.size(-1)
      # t = t.view(big_b_s//num_patches, num_patches, *t.size()[1:])
      z_ = z_.view(-1, emb_size) # flatten it, without changing anything
      h_ = h_.view(-1, emb_size)
      # loss_pkt += PKTClass.cosine_similarity_loss(z_,h_)
      _loss, neg_variance = PKTClass.cosine_similarity_loss_max_var(z_, h_)
      loss_pkt += _loss
      neg_variance_sum += neg_variance*variance_weight
  loss_pkt /= z.size(0) # normalize by batch size
  neg_variance_sum /= z.size(0)

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

def L2_PKT(z,h, **kwargs):
  """ Calculate the PKT loss. WIP
   
  :param z: the representation of the patches after being passed through 
   the Context Encoder and the Predictor
  :param h: the representation of the patches after being passed through
   the Target Encoder """

  loss_l2 = L2(z,h)
  loss_pkt = PKT(z,h)
  return loss_l2 + loss_pkt

def L2_PKT_scaled(z,h, **kwargs):
  """ Calculate the PKT loss while scaling the PKT part by a factor of 100 by default.
   
  :param z: the representation of the patches after being passed through 
   the Context Encoder and the Predictor
  :param h: the representation of the patches after being passed through
   the Target Encoder """
  pkt_scale = float(kwargs.get('pkt_scale', 100.)) # default to 1 if not present
  loss_l2 = L2(z,h)
  loss_pkt = PKT(z,h)
  return loss_l2 + loss_pkt * pkt_scale


def PKT_full(z,h, **kwargs):
  """ Calculate the PKT loss between all patches of all batches, at once.
   
  :param z: the representation of the patches after being passed through 
   the Context Encoder and the Predictor
  :param h: the representation of the patches after being passed through
   the Target Encoder """
  
  # we have [256, 20, 768]
  # we can view it as [256*20, 768] = [5120, 768]
  # and then perform PKT there
  # or take mean of second dimension and perform PKT over [256, 768]
  logger = logging.getLogger()

  w = kwargs.get('variance_weight', 1.)

  emb_size = z.size(-1)
  z_ = z.view(-1, emb_size) # [5120, 768]
  h_ = h.view(-1, emb_size)
  loss_pkt_full, neg_variance = PKTClass.cosine_similarity_loss_max_var(z_, h_)
  # logger.info('neg variance: %e, \n loss w/o max_var %e' % (neg_variance.item(), loss_pkt_full.item()))
  return loss_pkt_full + w*neg_variance

def L2_PKT_batch(z,h, **kwargs):
  alpha = kwargs.get('alpha', -1)
  if alpha == -1:
    return PKT_full(z,h) + L2(z,h) # normal behavior if alpha is absent
  return alpha*PKT_full(z,h) + (1-alpha) * L2(z,h)

def L2_PKT_chunks(z,h, **kwargs):
  """ Scale PKT after performing it in chunks of the patches """
  emb_size = z.size(-1) # take vector embedding dimension
  z_ = z.view(-1, emb_size) # transform from [256, 20, 768] into [5120, 768]
  h_ = h.view(-1, emb_size) 
  # create a random permutation
  rperm = torch.randperm(z_.size(-1)) # this yields an array of [1, 5, 0,... , 5119] random indices
  vsize = h_.size(0) # vector size
  step = 512 # hardcoded # vsize/10    # split sim matrix in x parts and run pkt in them
  
  assert h_.size(0) == z_.size(0), 'Different batch sizes between z,h ?'

  pkt_scale = kwargs.get('pkt_scale', 1.) # default to 1 if it fails
  # pkt_scale = 1.0e+3 # hardcoded for now

  loss_L2 = L2(z,h)
  loss_pkt = 0
  for i in range(0, vsize, step):
    # loss_pkt += PKTClass.cosine_similarity_loss(z_[rperm[i:i+step]],h_[rperm[i:i+step]])
    loss_pkt += PKTClass.cosine_similarity_loss(z_[i:i+step],h_[i:i+step])  
    # loss_L2 += L2(z_[i:i+step],h_[i:i+step])
  # logger.critical('loss inside PKT is : %s and pkt scale: %f' % (loss_pkt, pkt_scale))
  # logger.critical(pkt_scale)
  return (loss_pkt*pkt_scale + loss_L2)/(vsize/step)

def PKT_chunks(z,h, **kwargs):
  """ Scale PKT after performing it in chunks of the patches """
  emb_size = z.size(-1) # take vector embedding dimension
  z_ = z.view(-1, emb_size) # transform from [256, 20, 768] into [5120, 768]
  h_ = h.view(-1, emb_size) 
  # create a random permutation
  # rperm = torch.randperm(z_.size(-1)) # this yields an array of [1, 5, 0,... , 5119] random indices
  vsize = h_.size(0) # vector size
  step = 512 # hardcoded # vsize/10    # split sim matrix in x parts and run pkt in them
  
  assert h_.size(0) == z_.size(0), 'Different batch sizes between z,h ?'

  pkt_scale = kwargs.get('pkt_scale', 1.) # default to 1 if it fails
  # pkt_scale = 1.0e+3 # hardcoded for now

  # loss_L2 = L2(z,h)
  loss_pkt = 0
  for i in range(0, vsize, step):
    loss_pkt += PKTClass.cosine_similarity_loss(z_[i:i+step],h_[i:i+step])  
    
 
  return (loss_pkt*pkt_scale)/(vsize/step)

def L2_PKT_cross(z,h, **kwargs):  
  """L2 + PKT in chunks after performing it in chunks of the patches """
  emb_size = z.size(-1) # take vector embedding dimension
  z_ = z.view(-1, emb_size) # transform from [256, 20, 768] into [5120, 768]
  h_ = h.view(-1, emb_size) 
  # create a random permutation
  # rperm = torch.randperm(z_.size(-1)) # this yields an array of [1, 5, 0,... , 5119] random indices
  vsize = h_.size(0) # vector size
  step = 512 # hardcoded # vsize/10    # split sim matrix in x parts and run pkt in them
  
  assert h_.size(0) == z_.size(0), 'Different batch sizes between z,h ?'

  
  loss_L2 = L2(z,h)
  loss_pkt = 0
  mse = 0
  for i in range(0, vsize, step):
    loss_pkt_, mse_ = PKTClass.cosine_similarity_loss_cross_diag(z_[i:i+step],h_[i:i+step])  
    loss_pkt += loss_pkt_
    mse += mse_
  
  loss_pkt /= (vsize/step)
  mse /= (vsize/step)
    
 
  return (loss_pkt + loss_L2)/(vsize/step), mse
  

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

[5120, 768] @ [768, 5120] =  [5120, 5120] 
"""