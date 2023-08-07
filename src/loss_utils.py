import torch
from itertools import permutations

def permutation_loss(ref1, ref2, inf1, inf2, criterion):
    """
    Args:
        ref (List[torch.Tensor]): [(batch, ...), ...]
        inf (List[torch.Tensor]): [(batch, ...), ...]
        criterion (function): Loss function
        perm: (batch)
    Returns:
        torch.Tensor: (batch)
    """
    num_spk = len(ref1)

    def pair_loss(permutation):
        return sum(
            [(criterion(ref1[s], inf1[t]) + criterion(ref2[s], inf2[t]))*0.5 for s, t in enumerate(permutation)]
        ) / len(permutation)

    losses = torch.stack([pair_loss(p) for p in permutations(range(num_spk))], dim=1)
    loss, perm = torch.min(losses, dim=1)

    return loss


def permutation_loss2(ref1, ref2, inf1, inf2, criterion):
    
    ### keep dim ####
    
    """
    Args:
        ref (List[torch.Tensor]): [(batch, ...), ...]
        inf (List[torch.Tensor]): [(batch, ...), ...]
        criterion (function): Loss function
        perm: (batch)
    Returns:
        torch.Tensor: (batch)
    """
    num_spk = len(ref1)

    def pair_loss(permutation):
        return sum(
            [(criterion(ref1[s], inf1[t]) + criterion(ref2[s], inf2[t]))*0.5 for s, t in enumerate(permutation)]
        ) / len(permutation)

    losses = torch.stack([torch.mean(pair_loss(p),dim=1,keepdim=True) for p in permutations(range(num_spk))], dim=1)
    loss, perm = torch.min(losses, dim=1)

    return loss

def permutation_loss_weight(ref1, ref2, inf1, inf2, criterion, weight):
    """
    Args:
        ref (List[torch.Tensor]): [(batch, ...), ...]
        inf (List[torch.Tensor]): [(batch, ...), ...]
        criterion (function): Loss function
        perm: (batch)
    Returns:
        torch.Tensor: (batch)
    """
    num_spk = len(ref1)

    def pair_loss(permutation):
        return sum(
            [(criterion(ref1[s], inf1[t]) + criterion(ref2[s], inf2[t]))*0.5 for s, t in enumerate(permutation)]
        ) / len(permutation)
    
    losses = torch.stack([pair_loss(p) for p in permutations(range(num_spk))], dim=2)  #B,T,2
    losses_avg = torch.mean(losses, dim=1)  #B,2
    _, perm = torch.min(losses_avg, dim=1)
    
    losses = torch.stack([losses[i,:,v] for i,v in enumerate(perm)], dim=0)
    losses = losses * weight

    loss = torch.mean(losses)

    return loss


def permutation_loss_segment_weight(ref1, ref2, inf1, inf2, criterion, weight):
    """
    Args:
        ref (List[torch.Tensor]): [(batch, ...), ...]
        inf (List[torch.Tensor]): [(batch, ...), ...]
        criterion (function): Loss function
        perm: (batch)
    Returns:
        torch.Tensor: (batch)
    """
    num_spk = len(ref1)

    def pair_loss(permutation):
        return sum(
            [(criterion(ref1[s], inf1[t]) + criterion(ref2[s], inf2[t]))*0.5 for s, t in enumerate(permutation)]
        ) / len(permutation)

    losses = torch.stack([pair_loss(p) for p in permutations(range(num_spk))], dim=1)
    loss, perm = torch.min(losses, dim=1)
    
    
    batch_size = int(ref1[0].size(0))
    signal_len = int(ref1[0].size(1))
    
    ref1_seg = [item.reshape(batch_size, 4, signal_len//4).reshape(batch_size*4, signal_len//4) for item in ref1]
    ref2_seg = [item.reshape(batch_size, 4, signal_len//4).reshape(batch_size*4, signal_len//4) for item in ref2]
    inf1_seg = [item.reshape(batch_size, 4, signal_len//4).reshape(batch_size*4, signal_len//4) for item in inf1]
    inf2_seg = [item.reshape(batch_size, 4, signal_len//4).reshape(batch_size*4, signal_len//4) for item in inf2]
    

    all_loss = []
    
    for i,v in enumerate(perm):
        if v == 0:
            all_loss.append((criterion(ref1_seg[0][i*4:(i+1)*4,:], inf1_seg[0][i*4:(i+1)*4,:]) + 
                             criterion(ref2_seg[0][i*4:(i+1)*4,:], inf2_seg[0][i*4:(i+1)*4,:]) + 
                             criterion(ref1_seg[1][i*4:(i+1)*4,:], inf1_seg[1][i*4:(i+1)*4,:]) + 
                             criterion(ref2_seg[1][i*4:(i+1)*4,:], inf2_seg[1][i*4:(i+1)*4,:]))*0.25)
        else:
            all_loss.append((criterion(ref1_seg[0][i*4:(i+1)*4,:], inf1_seg[1][i*4:(i+1)*4,:]) + 
                             criterion(ref2_seg[0][i*4:(i+1)*4,:], inf2_seg[1][i*4:(i+1)*4,:]) + 
                             criterion(ref1_seg[1][i*4:(i+1)*4,:], inf1_seg[0][i*4:(i+1)*4,:]) + 
                             criterion(ref2_seg[1][i*4:(i+1)*4,:], inf2_seg[0][i*4:(i+1)*4,:]))*0.25)

        
    all_loss = torch.cat(all_loss, dim=0) #B*4
    loss = all_loss * weight
    loss = torch.mean(loss)
    
    return loss

def snr_loss(ref, inf):
    """
    :param ref: (block_num, samples)
    :param inf: (block_num, samples)
    :return: (Batch)
    """
    eps = 1e-8

    noise = ref - inf

    snr = 10 * torch.log10(ref.pow(2).sum(1) + eps) - 10 * torch.log10(noise.pow(2).sum(1) + eps)

    return -snr


def mse_loss(ref, inf):

    loss = nf.mse_loss(inf, ref, reduction='none').mean(dim=-1)
    return loss


def l1_loss(ref, inf):

    loss = nf.l1_loss(inf, ref, reduction='none').mean(dim=-1)
    return loss