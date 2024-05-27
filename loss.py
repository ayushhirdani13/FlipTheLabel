import torch
import torch.nn.functional as F

def flip_loss(y, label, flips, drop_rate):
    if flips is None or drop_rate is None:
        raise ValueError("flips and drop_rate must be provided for flip_loss")
    loss = F.binary_cross_entropy_with_logits(y, label, reduction='none')

    loss_mul = loss * label
    ind_sorted = torch.argsort(loss_mul.data)
    loss_sorted = loss[ind_sorted]

    remember_rate = 1 - drop_rate
    num_remember = int(remember_rate * len(loss_sorted))

    flip_inds = ind_sorted[num_remember:].cpu().data
    flips[flip_inds] = 0

    loss_update = F.binary_cross_entropy_with_logits(y, flips)

    return loss_update, flip_inds

def truncated_loss(y, label, drop_rate):
    if drop_rate is None:
        raise ValueError("drop_rate must be provided for truncated_loss")
    loss = F.binary_cross_entropy_with_logits(y, label, reduction='none')

    loss_mul = loss * label
    ind_sorted = torch.argsort(loss_mul.data)
    loss_sorted = loss[ind_sorted]

    remember_rate = 1 - drop_rate
    num_remember = int(remember_rate * len(loss_sorted))

    ind_update = ind_sorted[:num_remember]

    loss_update = F.binary_cross_entropy_with_logits(y[ind_update], label[ind_update])

    return loss_update