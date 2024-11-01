#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    tokenizer.py
# @Author:      Kuro
# @Time:        7/16/2024 10:48 AM
import functools

from transformers import AutoTokenizer

class TokenizerBase:
    def __init__(self, pretrained_tokenizer):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)

    def get_tokenizer(self):
        return self.tokenizer

class TokenizerMistral(TokenizerBase):
    def __init__(self, pretrained_tokenizer):
        super().__init__(pretrained_tokenizer)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    # Preprocess dataset with tokenizer
    def tokenize_examples(self, examples):
        tokenized_inputs = self.tokenizer(examples['text'], padding=True, truncation=True)
        tokenized_inputs['labels'] = examples['labels']
        return tokenized_inputs

    def preprocess_dataset(self, dataset):
        tokenized_ds = dataset.map(functools.partial(self.tokenize_examples), batched=True)
        tokenized_ds = tokenized_ds.with_format('torch')
        return tokenized_ds

