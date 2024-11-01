#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    train.py
# @Author:      Kuro
# @Time:        7/16/2024 10:48 AM
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class ClassificationTrainer:
    def __init__(self, model, tokenizer, optimizer, criterion, device='cuda', batch_size=32, epochs=10, lr=1e-3, max_length=512, validation=True):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.optimizer = optimizer(self.model.parameters(), lr=self.lr)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.validation = validation

    def train(self, x_train, y_train, dataset, x_val=None, y_val=None):
        train_loader = DataLoader(dataset(data=x_train, labels=y_train, tokenizer=self.tokenizer, max_length=self.max_length),
                                  batch_size=self.batch_size, shuffle=True)

        if self.validation and x_val is not None and y_val is not None:
            val_loader = DataLoader(dataset(data=x_val, labels=y_val, tokenizer=self.tokenizer, max_length=self.max_length),
                                    batch_size=self.batch_size, shuffle=False)

        self.model.to(self.device)

        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_one_epoch(train_loader)
            print(f"Epoch {epoch + 1}/{self.epochs} - Train loss: {train_loss} - Train acc: {train_acc}")

            if self.validation:
                val_loss, val_acc = self.validate(val_loader)
                print(f"Epoch {epoch + 1}/{self.epochs} - Val loss: {val_loss} - Val acc: {val_acc}")

    def train_one_epoch(self, train_loader):
        self.model.train()
        train_loss = 0
        train_acc = 0

        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(**inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            train_acc += (outputs.argmax(1) == labels).sum().item()

        return train_loss / len(train_loader), train_acc / len(train_loader.dataset)

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        val_acc = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                labels = labels.to(self.device)

                outputs = self.model(**inputs)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()
                val_acc += (outputs.argmax(1) == labels).sum().item()

        return val_loss / len(val_loader), val_acc / len(val_loader.dataset)
