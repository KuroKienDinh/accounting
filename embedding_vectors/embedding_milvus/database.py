#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    database.py
# @Author:      Kuro
# @Time:        7/16/2024 10:47 AM

from pymilvus import connections, utility, FieldSchema, Collection, CollectionSchema, DataType

from create_embedding import TransfomerEmbeddings


class MilvusCollection:
    def __init__(self, host, port: int, collection_name: str, dimension: int, index_param: dict, batch_size: int, batch_size_insert: int,
                 tokenizer, embedding_model, device):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dimension = dimension
        self.index_param = index_param
        self.batch_size = batch_size
        self.batch_size_insert = batch_size_insert

        self.embedding_model = TransfomerEmbeddings(tokenizer=tokenizer, model=embedding_model, dim=self.dimension, device=device)
        self.collection = None

    def connect(self):
        connections.connect(host=self.host, port=self.port)

    def drop_collection(self):
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)

    def create_collection(self, fields, field_name):
        self.collection.load()

    def insert(self, data):
        self.collection.insert(data)

    def upsert(self, data):
        self.collection.upsert(data)

    def delete(self, ids):
        self.collection.delete(ids)

    def flush(self):
        self.collection.flush()
