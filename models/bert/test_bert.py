#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    test_bert.py
# @Author:      Kuro
import unittest
from unittest import TestCase

from models.bert.model import BertForMaskedLMCustom


# @Time:        7/23/2024 4:16 PM
class TestBertForMaskedLMCustomModel(TestCase):

    def test_init(self):
        model = BertForMaskedLMCustom(pretrain_model_path='bert-base-uncased')
        self.assertIsNotNone(model.model)
        self.assertIsNotNone(model.optimizer)
        self.assertEqual(model.device, 'cuda')

    def test_class(self):
        model = BertForMaskedLMCustom(pretrain_model_path='bert-base-uncased')
        self.assertIsInstance(model.__class__(), tuple)  # Verify if it returns a tuple.
        self.assertEqual(len(model.__class__()), 2)  # Ensure it returns two elements (model, optimizer).

    def test_model_parameters(self):
        model = BertForMaskedLMCustom(pretrain_model_path='bert-base-uncased')
        params = list(model.model.parameters())
        self.assertGreater(len(params), 0)  # Check that the model has parameters.

    def test_optimizer_parameters(self):
        model = BertForMaskedLMCustom(pretrain_model_path='bert-base-uncased')
        optimizer_params = list(model.optimizer.param_groups)
        self.assertGreater(len(optimizer_params), 0)  # Check that the optimizer has a parameter group.

    def test_device(self):
        model = BertForMaskedLMCustom(pretrain_model_path='bert-base-uncased')
        # Check if model is on the correct device.
        self.assertEqual(next(model.model.parameters()).device, model.device)

if __name__ == '__main__':
    unittest.main()

