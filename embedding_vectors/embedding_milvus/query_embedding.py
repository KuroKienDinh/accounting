#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    query_embedding.py
# @Author:      Kuro
# @Time:        7/16/2024 10:42 AM

from typing import List

import torch
from pymilvus import MilvusClient

from create_embedding import TransfomerEmbeddings


class MilvusSearch:
    def __init__(self, tokenizer, embedding_model, dimension, device):
        self.embedding_model = TransfomerEmbeddings(tokenizer=tokenizer, model=embedding_model, dim=dimension, device=device)
        self.embedding_model.linear.load_state_dict(torch.load('../tool/item_rag_weights_207.pth'))
        self.client = MilvusClient(uri="http://localhost:19530")
        self.search_params = {"metric_type": "L2", "params": {"itopk_size": 16, "search_width": 16, "team_size": 8}}

    def search(self, collection_name, data: List[str], top_k: int = 10, filter="", output_fields=None):
        embedding = self.embedding_model.embed_documents(data)
        result = self.client.search(collection_name=collection_name, data=embedding, limit=top_k, filter=filter,
                                    output_fields=output_fields, search_params=self.search_params)
        return result
