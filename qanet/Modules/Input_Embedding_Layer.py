# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class Initialized_Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 1, stride = 1,
                 padding = 0, groups = 1, bias = True, relu=False):
        super().__init__()
        self.out = nn.Conv1d(in_channels, out_channels, kernel_size, stride = stride, padding = padding,
                             groups=groups, bias=bias)
        if relu is True:
            self.relu = True
            nn.init.kaiming_normal_(self.out.weight, nonlinearity = 'relu')
        else:
            self.relu = False
            nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        if self.relu is True:
            return F.relu(self.out(x))
        else:
            return self.out(x)

class Highway(nn.Module):
    def __init__(self, layer_num, dim):
        super().__init__()
        self.n = layer_num
        self.linear = nn.ModuleList([Initialized_Conv1d(dim, dim, bias = True) for _ in range(self.n)])
        self.gate = nn.ModuleList([Initialized_Conv1d(dim, dim, bias = True) for _ in range(self.n)])

    def forward(self, x):
        # x: shape [batch_size, hidden_size, length]
        dropout=0.1
        for i in range(self.n):
            gate = F.sigmoid(self.gate[i](x))
            nonlinear = self.linear[i](x)
            nonlinear = F.dropout(nonlinear, p=dropout, training=self.training)
            x = gate * nonlinear + (1-gate) * x
        return x

class Embedding(nn.Module):
    def __init__(self, word_emb_dim, char_emb_dim, model_dim, dropout_word = 0.1, dropout_char = 0.05):
        super().__init__()
        self.conv2d = nn.Conv2d(char_emb_dim, model_dim, kernel_size = (1,5), stride = 1, padding = 0, bias = True)
        nn.init.kaiming_normal_(self.conv2d.weight, nonlinearity = 'relu')
        self.conv1d = Initialized_Conv1d(word_emb_dim+model_dim, model_dim, bias=False)
        self.highway = Highway(2, model_dim)
        self.dropout_word = dropout_word
        self.dropout_char = dropout_char

    def forward(self, word_embedding, char_embedding):
        char_embedding = char_embedding.permute(0, 3, 1, 2)
        char_embedding = F.dropout(char_embedding, p = self.dropout_char, training = self.training)
        char_embedding = self.conv2d(char_embedding)
        char_embedding = F.relu(char_embedding)
        char_embedding, _ = torch.max(char_embedding, dim=3)

        word_embedding = F.dropout(word_embedding, p = self.dropout_word, training = self.training)
        word_embedding = word_embedding.transpose(1,2)
        embedding = torch.cat([char_embedding, word_embedding], dim=1)
        embedding = self.conv1d(embedding)
        embedding = self.highway(embedding)
        return embedding