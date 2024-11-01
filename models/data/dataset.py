#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    dataset.py
# @Author:      Kuro
# @Time:        7/18/2024 2:23 PM

class CustomDataset:
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        encoding = self.tokenizer.encode_plus(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0)


class ClassificationDataset(CustomDataset):
    def __init__(self, data, labels, tokenizer, max_length):
        super().__init__(data, tokenizer, max_length)
        self.labels = labels

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return encoding, label
