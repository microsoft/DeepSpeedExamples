# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch.utils.data import Dataset
from data_process import pickle_load_large_file


def collate(data):
    widsC, cidsC, widsQ, cidsQ, a_start, a_end, id, answerable = zip(*data)
    widsC = torch.Tensor(widsC).long()
    cidsC = torch.Tensor(cidsC).long()
    widsQ = torch.Tensor(widsQ).long()
    cidsQ = torch.Tensor(cidsQ).long()
    a_start = torch.Tensor(a_start).long()
    a_end = torch.Tensor(a_end).long()
    id = torch.Tensor(id).long()
    answerable = torch.Tensor(answerable).long()

    return widsC, cidsC, widsQ, cidsQ, a_start, a_end, id, answerable


class SQuADDataset(Dataset):

    def __init__(self, data_path):
        self.data = pickle_load_large_file(data_path)
        self.num = len(self.data)

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        return (self.data[item]['wids_context'],
                self.data[item]['cids_context'],
                self.data[item]['wids_question'],
                self.data[item]['cids_question'],
                self.data[item]['a_start'],
                self.data[item]['a_end'],
                self.data[item]['id'],
                self.data[item]['answerable'])
