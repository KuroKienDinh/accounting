#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    create_embedding.py
# @Author:      Kuro
# @Time:        7/16/2024 10:45 AM

from typing import List

import torch


class TransfomerEmbeddings:
    def __init__(self, tokenizer, model, dim, device):
        self.dim = dim
        self.device = device
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()
        self.model.to(device)
        self.linear = torch.nn.Linear(self.model.config.hidden_size, dim).to(device)

    def last_token_pool(self, last_hidden_states, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        batch_dict = self.tokenizer(texts, return_tensors="pt", max_length=self.dim, truncation=True, padding=True)
        batch_dict = batch_dict.to(self.device)
        outputs = self.model(**batch_dict)
        embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = self.linear(embeddings)
        return embeddings.cpu().detach().numpy().tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
