import torch

import logging

# logger = logging.getLogger()

def cosine_similarity_loss(output_net, target_net, eps=0.0000001):
    

    # Normalize each vector by its norm
    output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
    output_net = output_net / (output_net_norm + eps)
    output_net[output_net != output_net] = 0

    target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
    target_net = target_net / (target_net_norm + eps)
    target_net[target_net != target_net] = 0
    
    # Calculate the cosine similarity
    model_similarity = torch.mm(output_net, output_net.transpose(0, 1)) 
    target_similarity = torch.mm(target_net, target_net.transpose(0, 1))

    # cross = torch.mm(model_similarity, target_similarity) # cross sim matrix
    
    # sdiag = cross.diag() # <- tend to 1
    # sltri = cross.tril().sum()  # <- tend to 0 
    # ones_v = torch.ones_like(sdiag)

    # mse = torch.nn.functional.mse_loss(sdiag, ones_v) 

    # Scale cosine similarity to 0..1
    model_similarity = (model_similarity + 1.0) / 2.0
    target_similarity = (target_similarity + 1.0) / 2.0

    # Transform them into probabilities
    model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
    target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)

    # Calculate the KL-divergence
    loss = torch.mean(target_similarity * torch.log((target_similarity + eps) / (model_similarity + eps)))
    
    return loss


def cosine_similarity_loss_cross_diag(output_net, target_net, eps=0.0000001):

    # Normalize each vector by its norm
    output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
    output_net = output_net / (output_net_norm + eps)
    output_net[output_net != output_net] = 0

    target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
    target_net = target_net / (target_net_norm + eps)
    target_net[target_net != target_net] = 0

    cross = torch.mm(output_net, target_net) # cross sim matrix

    # Calculate the cosine similarity
    model_similarity = torch.mm(output_net, output_net.transpose(0, 1)) 
    target_similarity = torch.mm(target_net, target_net.transpose(0, 1))


    sdiag = cross.diag() # <- tend to 1
    # sltri = cross.tril().sum()  # <- tend to 0
    ones_v = torch.ones_like(sdiag)

    mse = torch.nn.functional.mse_loss(sdiag, ones_v) 

    # Scale cosine similarity to 0..1
    model_similarity = (model_similarity + 1.0) / 2.0
    target_similarity = (target_similarity + 1.0) / 2.0

    # Transform them into probabilities
    model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
    target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)

    # Calculate the KL-divergence
    loss = torch.mean(target_similarity * torch.log((target_similarity + eps) / (model_similarity + eps)))

    return loss, mse


def cross_similarity_loss(output_net, target_net, eps=0.0000001):
    # Normalize each vector by its norm
    output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
    output_net = output_net / (output_net_norm + eps)
    output_net[output_net != output_net] = 0

    target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
    target_net = target_net / (target_net_norm + eps)
    target_net[target_net != target_net] = 0

    # Calculate the cosine similarity
    cross = torch.mm(output_net, target_net) # cross sim matrix
    
    sdiag = -cross.diag().sum() # <- tend to 1
    sltri = cross.tril().sum()  # <- tend to 0 

    w = 1.e-1
    return sdiag + w*sltri


# take inter-image similarity 
def cosine_similarity_loss_inter(output_net, target_net, output_net_2, target_net_2, eps=0.0000001):
    # Normalize each vector by its norm
    output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
    output_net = output_net / (output_net_norm + eps)
    output_net[output_net != output_net] = 0

    target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
    target_net = target_net / (target_net_norm + eps)
    target_net[target_net != target_net] = 0

    # Normalize each vector by its norm a second time for another image
    output_net_norm_2 = torch.sqrt(torch.sum(output_net_2 ** 2, dim=1, keepdim=True))
    output_net_2 = output_net_2 / (output_net_norm_2 + eps)
    output_net_2[output_net_2 != output_net_2] = 0

    target_net_norm_2 = torch.sqrt(torch.sum(target_net_2 ** 2, dim=1, keepdim=True))
    target_net_2 = target_net_2 / (target_net_norm_2 + eps)
    target_net_2[target_net_2 != target_net_2] = 0

    # Calculate the cosine similarity
    model_similarity = torch.mm(output_net, output_net_2.transpose(0, 1))
    target_similarity = torch.mm(target_net, target_net_2.transpose(0, 1))

    # Scale cosine similarity to 0..1
    model_similarity = (model_similarity + 1.0) / 2.0
    target_similarity = (target_similarity + 1.0) / 2.0

    # Transform them into probabilities
    model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
    target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)

    # Calculate the KL-divergence
    loss = torch.mean(target_similarity * torch.log((target_similarity + eps) / (model_similarity + eps)))

    return loss


def get_similarity_matrices(output_net, target_net, eps=0.0000001):
    """ Return similarity matrices for model and target. Should 
    be symmetric"""
    # Normalize each vector by its norm
    # print('panw',output_net)
    output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
    output_net = output_net / (output_net_norm + eps)
    output_net[output_net != output_net] = 0

    target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
    target_net = target_net / (target_net_norm + eps)
    target_net[target_net != target_net] = 0
    
    cross = torch.mm(output_net, target_net) # cross sim matrix

    # Calculate the cosine similarity
    # print('katw',output_net)
    model_similarity = torch.mm(output_net, output_net.transpose(0, 1))
    target_similarity = torch.mm(target_net, target_net.transpose(0, 1))

    # Scale cosine similarity to 0..1
    model_similarity = (model_similarity + 1.0) / 2.0
    target_similarity = (target_similarity + 1.0) / 2.0
    # cross = (cross + 1.) / 2.


    # # Transform them into probabilities
    # model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
    # target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)

    return model_similarity, target_similarity, cross


def get_similarity_distribution(output_net, target_net, eps=0.0000001):
    """ Return similarities for model and target sim matrices. """
    # Normalize each vector by its norm
    # print('panw',output_net)
    output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
    output_net = output_net / (output_net_norm + eps)
    output_net[output_net != output_net] = 0

    target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
    target_net = target_net / (target_net_norm + eps)
    target_net[target_net != target_net] = 0

    # Calculate the cosine similarity
    # print('katw',output_net)
    model_similarity = torch.mm(output_net, output_net.transpose(0, 1))
    target_similarity = torch.mm(target_net, target_net.transpose(0, 1))

    # Scale cosine similarity to 0..1
    model_similarity = (model_similarity + 1.0) / 2.0
    target_similarity = (target_similarity + 1.0) / 2.0

    return model_similarity.view(-1), target_similarity.view(-1) # maybe get only upper triangular with .triu()?
    
    # # Transform them into probabilities
    # model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
    # target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)

    # # Calculate the KL-divergence
    # loss = torch.mean(target_similarity * torch.log((target_similarity + eps) / (model_similarity + eps)))

    # return loss