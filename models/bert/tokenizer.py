#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    tokenizer.py
# @Author:      Kuro
# @Time:        7/16/2024 10:48 AM
from tokenizers.implementations import BertWordPieceTokenizer
from transformers import AutoTokenizer


class BertWordPieceTokenizerCustom:
    def __init__(self):
        self.tokenizer = BertWordPieceTokenizer()

    def __class__(self):
        return self.tokenizer

class TokenizerCustom:
    def __init__(self, pretrained_tokenizer):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)

    def __class__(self):
        return self.tokenizer

