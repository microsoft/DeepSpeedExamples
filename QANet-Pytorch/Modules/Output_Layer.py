# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Input_Embedding_Layer import Initialized_Conv1d

def mask_logits(target, mask):
    mask = mask.type(torch.float32)
    return target * mask + (1 - mask) * (-1e30)

class Pointer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w1 = Initialized_Conv1d(d_model*2, 1)
        self.w2 = Initialized_Conv1d(d_model*2, 1)

    def forward(self, M1, M2, M3, mask):
        X1 = torch.cat([M1, M2], dim=1)
        X2 = torch.cat([M1, M3], dim=1)
        Y1 = mask_logits(self.w1(X1).squeeze(), mask)
        Y2 = mask_logits(self.w2(X2).squeeze(), mask)
        return Y1, Y2