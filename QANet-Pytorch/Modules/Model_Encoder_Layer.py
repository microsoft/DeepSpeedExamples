# -*- coding: utf-8 -*-
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .Input_Embedding_Layer import Initialized_Conv1d

def mask_logits(target, mask):
    mask = mask.type(torch.float32)
    return target * mask + (1 - mask) * (-1e30)

def PosEncoder(x, min_timescale=1.0, max_timescale=1.0e4):
    x = x.transpose(1, 2)
    length = x.size()[1]
    channels = x.size()[2]
    signal = get_timing_signal(length, channels, min_timescale, max_timescale)
    return (x + signal.to(x.get_device())).transpose(1, 2)

def get_timing_signal(length, channels,
                      min_timescale=1.0, max_timescale=1.0e4):
    position = torch.arange(length).type(torch.float32)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales).type(torch.float32) * -log_timescale_increment)
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim = 1)
    m = nn.ZeroPad2d((0, (channels % 2), 0, 0))
    signal = m(signal)
    signal = signal.view(1, length, channels)
    return signal

class Depthwise_Seperable_Conv(nn.Module):
    def __init__(self, num_inchannel, num_outchannel, k, bias = True):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(num_inchannel, num_inchannel, kernel_size = k, groups = num_inchannel,
                                        padding = k//2, bias = False)
        self.pointwise_conv = nn.Conv1d(num_inchannel, num_outchannel, kernel_size = 1, padding = 0, bias=bias)

    def forward(self, x):
        return F.relu(self.pointwise_conv(self.depthwise_conv(x)))

class Self_Attention(nn.Module):
    def __init__(self, model_dim, num_heads, dropout):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.mem_conv = Initialized_Conv1d(in_channels = model_dim, out_channels = model_dim*2, kernel_size = 1,
                                           relu = False, bias = False)
        self.query_conv = Initialized_Conv1d(in_channels = model_dim, out_channels = model_dim, kernel_size = 1,
                                             relu = False, bias = False)

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)

    def forward(self, queries, mask):
        memory = queries

        memory = self.mem_conv(memory)
        query = self.query_conv(queries)
        memory = memory.transpose(1, 2)
        query = query.transpose(1, 2)
        Q = self.split_last_dim(query, self.num_heads)
        K, V = [self.split_last_dim(tensor, self.num_heads) for tensor in torch.split(memory, self.model_dim, dim=2)]

        key_depth_per_head = self.model_dim // self.num_heads
        Q *= key_depth_per_head ** -0.5
        x = self.dot_product_attention(Q, K, V, mask=mask)
        return self.combine_last_two_dim(x.permute(0, 2, 1, 3)).transpose(1, 2)

    def dot_product_attention(self, q, k ,v, bias = False, mask = None):
        logits = torch.matmul(q, k.permute(0, 1, 3, 2))
        if bias:
            logits += self.bias
        if mask is not None:
            shapes = [x if x != None else -1 for x in list(logits.size())]
            mask = mask.view(shapes[0], 1, 1, shapes[-1])
            logits = mask_logits(logits, mask)
        weights = F.softmax(logits, dim = -1)
        # dropping out the attention links for each of the heads
        weights = F.dropout(weights, p = self.dropout, training = self.training)
        return torch.matmul(weights, v)

    def split_last_dim(self, x, n):
        old_shape = list(x.size())
        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        ret = x.view(new_shape)
        return ret.permute(0, 2, 1, 3)

    def combine_last_two_dim(self, x):
        old_shape = list(x.size())
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a*b if a and b else None]
        ret = x.contiguous().view(new_shape)
        return ret

class EncoderBlock_Model(nn.Module):
    def __init__(self, num_conv, model_dim, num_heads, k, dropout):
        super().__init__()
        self.convs = nn.ModuleList([Depthwise_Seperable_Conv(model_dim, model_dim, k) for _ in range(num_conv)])
        self.self_att = Self_Attention(model_dim, num_heads, dropout = dropout)
        self.num_conv = num_conv
        self.dropout = dropout
        self.norm_C = nn.ModuleList([nn.LayerNorm(model_dim) for _ in range(num_conv)])
        self.norm_1 = nn.LayerNorm(model_dim)
        self.norm_2 = nn.LayerNorm(model_dim)
        self.FFN_1 = Initialized_Conv1d(model_dim, model_dim, relu=True, bias=True)
        self.FFN_2 = Initialized_Conv1d(model_dim, model_dim, bias=True)


    def forward(self, x, mask, l, blks):
        total_layers = (self.num_conv + 1) * blks
        dropout = self.dropout
        out = PosEncoder(x)
        for i, conv in enumerate(self.convs):
            res = out
            out = self.norm_C[i](out.transpose(1,2)).transpose(1,2)
            if (i) % 2 == 0:
                out = F.dropout(out, p = dropout, training = self.training)
            out = conv(out)
            out = self.layer_dropout(out, res, dropout * float(l)/total_layers)
            l += 1
        res = out
        out = self.norm_1(out.transpose(1,2)).transpose(1,2)
        out = F.dropout(out, p=dropout, training=self.training)
        out = self.self_att(out, mask)
        out = self.layer_dropout(out, res, dropout * float(l)/total_layers)
        l += 1
        res = out
        out = self.norm_2(out.transpose(1,2)).transpose(1,2)
        out = F.dropout(out, p=dropout, training=self.training)
        out = self.FFN_1(out)
        out = self.FFN_2(out)
        out = self.layer_dropout(out, res, dropout * float(l) / total_layers)
        return out

    def layer_dropout(self, inputs, residual, dropout):
        if self.training == True:
            pred = torch.empty(1).uniform_(0,1) < dropout
            if pred:
                return residual
            else:
                return F.dropout(inputs, dropout, training=self.training) + residual
        else:
            return inputs + residual
