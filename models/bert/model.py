#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    model.py
# @Author:      Kuro
# @Time:        7/16/2024 10:48 AM

from transformers import BertForMaskedLM
from torch.optim import AdamW
from transformers import AutoModel
class BertForMaskedLMCustom:
    def __init__(self, pretrain_model_path, device='cuda'):
        self.model = BertForMaskedLM.from_pretrained(pretrain_model_path).to(device)
        self.device = device
        self.optimizer = AdamW(self.model.parameters(), lr=1e-4)

    def __class__(self):
        return self.model, self.optimizer

class BertCustom:
    def __init__(self, pretrained_model, device='cuda'):
        self.model = AutoModel.from_pretrained(pretrained_model).to(device)
        self.device = device
        self.optimizer = AdamW(self.model.parameters(), lr=1e-4)

    def __class__(self):
        return self.model, self.optimizer

