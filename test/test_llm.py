#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    test_llm.py
# @Author:      Kuro
# @Time:        7/28/2024 3:31 PM


import pandas as pd
import torch
from models.llm.model import GemmaModel
from models.llm.train import ClassificationTrainer
from models.data.dataset import ClassificationDataset
from models.llm.tokenizer import TokenizerGemma
from torch import nn

data_path = "../dataset"
food_path = ["../dataset/food_beverage/Food/Food.xlsx"]
beverage_path = ["../dataset/food_beverage/Beverage/BEVERAGE.xlsx", "../dataset/food_beverage/Beverage/Getraenke.xlsx", "../dataset/food_beverage/Beverage/Getranke.xlsx", "../dataset/food_beverage/Beverage/Getränke.xlsx"]

df_food = pd.DataFrame()
df_beverage = pd.DataFrame()

for path in food_path:
    df_food = pd.concat([df_food, pd.read_excel(path)], ignore_index=True)
for path in beverage_path:
    df_beverage = pd.concat([df_beverage, pd.read_excel(path)], ignore_index=True)

df_food['class'] = "food"
df_beverage['class'] = "beverage"

df = pd.concat([df_food, df_beverage], ignore_index=True)
df = df[["bez", "class"]]
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

class_mapping = {'food': 0, 'beverage': 1}
df['class'] = df['class'].map(class_mapping)

df = df.sample(frac=1).reset_index(drop=True)

test_size = 0.2

train_df = df.iloc[:int(len(df)*(1-test_size))]
test_df = df.iloc[int(len(df)*(1-test_size)):]

X_train, y_train = train_df['bez'].values, train_df['class'].values
X_test, y_test = test_df['bez'].values, test_df['class'].values

pretrained_model = "BAAI/bge-multilingual-gemma2"
device = "cuda"
n_classes = 2

optimizer = torch.optim.Adam
criterion = nn.CrossEntropyLoss()

max_length = 128
lr = 1e-5
batch_size = 32
epochs = 10

tokenizer = TokenizerGemma(pretrained_model).get_tokenizer()
model = GemmaModel(pretrained_model=pretrained_model, n_classes=n_classes, device=device).get_model()
trainer = ClassificationTrainer(model=model, tokenizer=tokenizer, optimizer=optimizer, criterion=criterion, device=device, batch_size=batch_size, epochs=epochs, lr=lr, max_length=max_length)
trainer.train(X_train, y_train, ClassificationDataset, X_test, y_test)

