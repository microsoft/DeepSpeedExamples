# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from Modules.Input_Embedding_Layer import Embedding, Initialized_Conv1d
from Modules.Embedding_Encoder_Layer import EncoderBlock
from Modules.Context_Query_Attention_Layer import CQAttention
from Modules.Model_Encoder_Layer import EncoderBlock_Model
from Modules.Output_Layer import Pointer


class QANet_Unanswerable(nn.Module):

    def __init__(self, word_emb_matrix, char_emb_matrix, model_dim, num_heads = 1,
                 train_char_emb = False, pad = 0, dropout = 0.1):
        super().__init__()
        if train_char_emb:
            self.char_emb = nn.Embedding.from_pretrained(char_emb_matrix)
        else:
            self.char_emb = nn.Embedding.from_pretrained(char_emb_matrix, freeze=False)
        self.word_emb = nn.Embedding.from_pretrained(word_emb_matrix)
        word_emb_dim = word_emb_matrix.shape[1]
        char_emb_dim = char_emb_matrix.shape[1]
        self.embedding = Embedding(word_emb_dim, char_emb_dim, model_dim)
        self.num_heads = num_heads
        self.PAD = pad
        self.dropout = dropout
        self.emb_encoder = EncoderBlock(num_conv=4, model_dim = model_dim, num_heads = num_heads, k = 7, dropout = 0.1)
        self.cq_att = CQAttention(model_dim = model_dim)
        self.cq_resizer = Initialized_Conv1d(model_dim * 4, model_dim)
        self.model_encoder_blks = nn.ModuleList([EncoderBlock_Model(num_conv = 2, model_dim = model_dim,
                                                         num_heads = num_heads , k = 5, dropout=0.1) for _ in range(7)])
        self.out = Pointer(model_dim)
    def forward(self, widsC, cidsC, widsQ, cidsQ):
        maskC = (torch.ones_like(widsC) * self.PAD != widsC).float()
        maskQ = (torch.ones_like(widsQ) * self.PAD != widsQ).float()
        wC_emb = self.word_emb(widsC)
        cC_emb = self.char_emb(cidsC)
        wQ_emb = self.word_emb(widsQ)
        cQ_emb = self.char_emb(cidsQ)
        C = self.embedding(wC_emb, cC_emb)
        Q = self.embedding(wQ_emb, cQ_emb)
        Ce = self.emb_encoder(C, maskC, 1, 1)
        Qe = self.emb_encoder(Q, maskQ, 1, 1)
        X = self.cq_att(Ce, Qe, maskC, maskQ)
        M0 = self.cq_resizer(X)
        M0 = F.dropout(M0, p=self.dropout, training = self.training)
        for i, blk in enumerate(self.model_encoder_blks):
            M0 = blk(M0, maskC, i*(2+2)+1, 7)
        M1 = M0
        for i, blk in enumerate(self.model_encoder_blks):
             M0 = blk(M0, maskC, i*(2+2)+1, 7)
        M2 = M0
        M0 = F.dropout(M0, p=self.dropout, training=self.training)
        for i, blk in enumerate(self.model_encoder_blks):
            M0 = blk(M0, maskC, i * (2+2) + 1, 7)
        M3 = M0
        p1, p2 = self.out(M1, M2, M3, maskC)
        return p1, p2

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('Trainable parameters:', params)

