import torch
import numpy as np
from einops import rearrange

def positionalencoding3d(d_model, height, width, depth, type='sinusoidal'):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :param depth: depth of the positions
    :return: d_model*height*width position matrix, shape [H*W*D, C], value range [-1,1]
    """
    if type == 'sinusoidal':
        d_model_interv = int(np.ceil(d_model / 6) * 2)
        if d_model_interv % 2:
            d_model_interv += 1
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model_interv, 2).float() / d_model_interv))
        pos_x = torch.arange(height).type(inv_freq.type())
        pos_y = torch.arange(width).type(inv_freq.type())
        pos_z = torch.arange(depth).type(inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1).unsqueeze(1)
        emb_y = get_emb(sin_inp_y).unsqueeze(1)
        emb_z = get_emb(sin_inp_z)
        emb = torch.zeros(height, width, depth, d_model_interv * 3)
        emb[:, :, :, : d_model_interv] = emb_x
        emb[:, :, :, d_model_interv : 2 * d_model_interv] = emb_y
        emb[:, :, :, 2 * d_model_interv :] = emb_z
        emb = rearrange(emb, 'h w d c -> (h w d) c')[:,:d_model]
    elif type == 'zero':
        emb = torch.zeros(height * width * depth, d_model)
    return emb


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)