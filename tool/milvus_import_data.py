#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    milvus_import_data.py
# @Author:      kien.tadinh
# @Time:        5/2/2024 3:23 PM

import os
import warnings
from typing import List

import pandas as pd
import torch
from pymilvus import connections, utility, FieldSchema, Collection, CollectionSchema, DataType
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")


class ModelEmbeddings:
    def __init__(self, model_name: str, dim: int, device: str = "cuda"):
        self.dim = dim
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(device)
        self.linear = torch.nn.Linear(self.model.config.hidden_size, dim).to(device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        inputs = self.tokenizer(texts, return_tensors="pt", max_length=self.dim, truncation=True, padding=True)
        inputs = inputs.to(self.device)
        embeddings = self.model(**inputs)[0][:, 0, :]
        embeddings = self.linear(embeddings)
        return embeddings.cpu().detach().numpy().tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class MilvusEntity:
    def __init__(self, columns):
        self.columns = columns

    def create_entities(self):
        return pd.DataFrame(columns=self.columns)


class MilvusCollection:
    def __init__(self, host: str, port: int, collection_name: str, dimension: int, index_param: dict, batch_size: int, batch_size_insert: int,
                 embedding_model):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dimension = dimension
        self.index_param = index_param
        self.batch_size = batch_size
        self.batch_size_insert = batch_size_insert

        self.embedding_model = ModelEmbeddings(model_name=embedding_model, dim=self.dimension)
        self.collection = None

    def connect(self):
        connections.connect(host=self.host, port=self.port)

    def drop_collection(self):
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)

    def create_collection(self):
        if self.collection_name == "cisbox":
            fields = [
                FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name='item_name', dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name='account_name', dtype=DataType.VARCHAR, max_length=250),
                FieldSchema(name='group', dtype=DataType.INT16),
                FieldSchema(name='count_item_duplicate', dtype=DataType.INT32),
                FieldSchema(name='count_account_item', dtype=DataType.INT32),
                FieldSchema(name='item_name_embedding', dtype=DataType.FLOAT_VECTOR, dim=self.dimension)
            ]
            schema = CollectionSchema(fields=fields)
            self.collection = Collection(name=self.collection_name, schema=schema)
            self.collection.create_index(field_name="item_name_embedding", index_params=self.index_param)

        elif self.collection_name == "cisbox_account":
            fields = [
                FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name='account_name', dtype=DataType.VARCHAR, max_length=250),
                FieldSchema(name='group', dtype=DataType.INT16),
                FieldSchema(name='account_name_embedding', dtype=DataType.FLOAT_VECTOR, dim=self.dimension)
            ]
            schema = CollectionSchema(fields=fields)
            self.collection = Collection(name=self.collection_name, schema=schema)
            self.collection.create_index(field_name="account_name_embedding", index_params=self.index_param)
        self.collection.load()

    def insert(self, data):
        self.collection.insert(data)

    def upsert(self, data):
        self.collection.upsert(data)

    def delete(self, ids):
        self.collection.delete(ids)

    def flush(self):
        self.collection.flush()


def preprocessing_df(dataframe, group):
    dataframe['group'] = int(group)
    dataframe['count_bezeichnung'] = dataframe['bezeichnung'].map(dataframe['bezeichnung'].value_counts())
    dataframe['count_artikelbezeichnung1'] = dataframe.groupby(['artikelbezeichnung1', 'bezeichnung', 'group'])['artikelbezeichnung1'].transform(
        'count')
    dataframe = dataframe[['bezeichnung', 'artikelbezeichnung1', 'group', 'count_artikelbezeichnung1', 'count_bezeichnung']]
    dataframe.rename(columns={'bezeichnung': 'account_name', 'artikelbezeichnung1': 'item_name', 'count_artikelbezeichnung1': 'count_item_duplicate',
                              'count_bezeichnung': 'count_account_item'}, inplace=True)
    dataframe.dropna(inplace=True)
    dataframe.drop_duplicates(inplace=True, subset=['account_name', 'item_name', 'group'])
    dataframe.reset_index(drop=True, inplace=True)
    return dataframe


def importing_one_group_items(collection):
    columns = ['item_name', 'account_name', 'group', 'count_item_duplicate', 'count_account_item', 'item_name_embedding']
    vector_entities = MilvusEntity(columns=columns)
    entities = vector_entities.create_entities()
    total_records = 0
    df = pd.read_parquet('../data/master/groups/207/207.masterdata.parquet')
    df = preprocessing_df(df, 207)
    for i in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[i: i + BATCH_SIZE]
        item_name_embeddings = embedding_model.embed_documents(batch['item_name'].tolist())
        batch['item_name_embedding'] = item_name_embeddings
        entities = pd.concat([entities, batch], ignore_index=True, sort=False)
        if len(entities['item_name']) >= BATCH_SIZE_INSERT:
            total_records += len(entities['item_name'])
            collection.insert(entities.values.T.tolist())
            entities = vector_entities.create_entities()

    if not entities.empty:
        total_records += len(entities['item_name'])
        collection.insert(entities.values.T.tolist())

    collection.flush()
    torch.save(embedding_model.linear.state_dict(), 'item_rag_weights_207.pth')


def importing_group_items(collection):
    columns = ['item_name', 'account_name', 'group', 'count_item_duplicate', 'count_account_item', 'item_name_embedding']
    vector_entities = MilvusEntity(columns=columns)
    entities = vector_entities.create_entities()
    for data_type in ['model_cisbox_100', 'model_HGK_50']:
        total_records = 0
        for group in tqdm(os.listdir(f"../data/{data_type}")):
            df = pd.read_parquet(f'../data/{data_type}/{group}/{group}.masterdata.parquet')
            if df.empty:
                continue
            df = preprocessing_df(df, group)
            for i in range(0, len(df), BATCH_SIZE):
                batch = df.iloc[i: i + BATCH_SIZE]
                item_name_embeddings = embedding_model.embed_documents(batch['item_name'].tolist())
                batch['item_name_embedding'] = item_name_embeddings
                entities = pd.concat([entities, batch], ignore_index=True, sort=False)
                if len(entities['item_name']) >= BATCH_SIZE_INSERT:
                    total_records += len(entities['item_name'])
                    collection.insert(entities.values.T.tolist())
                    entities = vector_entities.create_entities()

            if total_records > 5000:
                break

    if not entities.empty:
        total_records += len(entities['item_name'])
        collection.insert(entities.values.T.tolist())

    collection.flush()
    torch.save(embedding_model.linear.state_dict(), 'item_rag_weights_207.pth')


# def importing_group_items():
#     columns = ['item_name', 'account_name', 'group', 'count_item_duplicate', 'count_account_item', 'item_name_embedding']
#     vector_entities = MilvusEntity(columns=columns)
#     entities = vector_entities.create_entities()
#     total_records = 0
#     for group in tqdm(os.listdir(f"../data/master/groups")):
#         df = pd.read_parquet(f'../data/master/groups/{group}/{group}.masterdata.parquet')
#         if df.empty:
#             continue
#         df = preprocessing_df(df, group)
#         for i in range(0, len(df), BATCH_SIZE):
#             batch = df.iloc[i: i + BATCH_SIZE]
#             item_name_embeddings = embedding_model.embed_documents(batch['item_name'].tolist())
#             batch['item_name_embedding'] = item_name_embeddings
#             entities = pd.concat([entities, batch], ignore_index=True, sort=False)
#             if len(entities['item_name']) >= BATCH_SIZE_INSERT:
#                 total_records += len(entities['item_name'])
#                 db_collection.insert(entities.values.T.tolist())
#                 entities = vector_entities.create_entities()
#
#     if not entities.empty:
#         total_records += len(entities['item_name'])
#         db_collection.insert(entities.values.T.tolist())
#
#     db_collection.flush()
#     torch.save(embedding_model.linear.state_dict(), 'item_rag_weights.pth')


def importing_account_name(collection):
    columns = ['account_name', 'group', 'account_name_embedding']
    vector_entities = MilvusEntity(columns=columns)
    entities = vector_entities.create_entities()
    for data_type in ['model_cisbox_100', 'model_HGK_50']:
        total_records = 0
        for group in tqdm(os.listdir(f"../data/{data_type}")):
            if not os.path.exists(f'../data/{data_type}/{group}/{group}.konto.parquet'):
                continue
            df = pd.read_parquet(f'../data/{data_type}/{group}/{group}.konto.parquet')
            if df.empty:
                continue
            df = df[['bezeichnung']]
            df.columns = ['account_name']
            df.dropna(inplace=True)
            df.drop_duplicates(inplace=True)
            df.reset_index(drop=True, inplace=True)
            account_name_embeddings = embedding_model.embed_documents(df['account_name'].tolist())
            df['account_name_embedding'] = account_name_embeddings
            df['group'] = int(group)
            entities = pd.concat([entities, df], ignore_index=True, sort=False)
            if len(entities['account_name']) >= BATCH_SIZE_INSERT:
                total_records += len(entities['account_name'])
                collection.insert(entities.values.T.tolist())
                entities = vector_entities.create_entities()

    if not entities.empty:
        total_records += len(entities['account_name'])
        collection.insert(entities.values.T.tolist())

    collection.flush()
    torch.save(embedding_model.linear.state_dict(), 'account_rag_weights.pth')


if __name__ == "__main__":
    ####### ----------------- Config ----------------- #######
    DEVICE = 'cuda'
    EMBEDDING_MODEL = "bert-base-german-cased"

    HOST = 'localhost'
    PORT = 19530

    DIMENSION = 256

    INDEX_PARAM = {'metric_type': 'COSINE',
                   'index_type': 'GPU_CAGRA',
                   'params': {'intermediate_graph_degree': 64, 'graph_degree': 32, 'cache_dataset_on_device': 'false'}
                   }
    BATCH_SIZE = 32
    BATCH_SIZE_INSERT = 1000
    COLLECTION_NAME = 'cisbox'  # cisbox or cisbox_account
    ####### ----------------- Config embedding ----------------- #######
    embedding_model = ModelEmbeddings(model_name=EMBEDDING_MODEL, dim=DIMENSION)
    ####### ----------------- Config Milvus ----------------- #######
    # Connect to Milvus Database
    db_collection = MilvusCollection(host=HOST, port=PORT, collection_name=COLLECTION_NAME, dimension=DIMENSION, index_param=INDEX_PARAM,
                                     batch_size=BATCH_SIZE, batch_size_insert=BATCH_SIZE_INSERT, embedding_model=EMBEDDING_MODEL)
    db_collection.connect()
    db_collection.drop_collection()
    db_collection.create_collection()
    ####### ----------------- Import Data ----------------- #######
    if COLLECTION_NAME == 'cisbox':
        # importing_group_items(db_collection)
        importing_one_group_items(db_collection)
    elif COLLECTION_NAME == 'cisbox_account':
        importing_account_name(db_collection)
