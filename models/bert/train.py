#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    train.py
# @Author:      Kuro
# @Time:        7/16/2024 10:48 AM
from tqdm import tqdm
from transformers import BertForMaskedLM, AdamW


class BertForMaskedLMTrainer:
    def __init__(self, model, optimizer, device='cuda', lr=1e-5):
        self.model = model
        self.device = device
        self.optimizer = optimizer

    def train(self, epoch, train_loader, do_eval=False, eval_loader=None):
        for epoch in tqdm(range(epoch)):
            self.model.train()
            for input_ids, attention_mask in train_loader:
                input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            if do_eval:
                eval_loss = self.eval(eval_loader)
                print(f'Epoch: {epoch}, Train Loss: {loss.item()}, Eval Loss: {eval_loss.item()}')
            else:
                print(f'Epoch: {epoch}, Train Loss: {loss.item()}')

    def eval(self, eval_loader):
        self.model.eval()
        for input_ids, attention_mask in eval_loader:
            input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            return loss

    def save(self, save_path):
        self.model.save_pretrained(save_path)

    def load(self, load_path):
        self.model = BertForMaskedLM.from_pretrained(load_path).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=1e-4)
        return self.model, self.optimizer

    def predict(self, text):
        pass


class BertWordPieceTokenizerTrainer:
    def __init__(self, tokenizer, pretrained_tokenizer, max_length):
        self.tokenizer = tokenizer
        self.pretrained_tokenizer = pretrained_tokenizer
        self.max_length = max_length

    def train(self, data):
        self.tokenizer.train_from_iterator(data, vocab_size=self.pretrained_tokenizer.vocab_size)
        self.tokenizer.enable_truncation(self.max_length)

    def save(self, save_path):
        self.tokenizer.save(save_path)
