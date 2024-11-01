#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    config.py
# @Author:      Kuro
# @Time:        7/16/2024 10:48 AM
import torch
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import BitsAndBytesConfig


class LLMConfig:
    def __init__(self):
        self.quantization = self.get_quantization_config()
        self.lora = self.get_lora_config()

    def get_quantization_config(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,  # enable 4-bit quantization
            bnb_4bit_quant_type='nf4',  # information theoretically optimal dtype for normally distributed weights
            bnb_4bit_use_double_quant=True,  # quantize quantized weights
            bnb_4bit_compute_dtype=torch.bfloat16  # optimized fp format for ML
        )
        return quantization_config

    def get_lora_config(self):
        lora_config = LoraConfig(
            r=16,  # the dimension of the low-rank matrices
            lora_alpha=8,  # scaling factor for LoRA activations vs pre-trained weight activations
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            lora_dropout=0.05,  # dropout probability of the LoRA layers
            bias='none',  # whether to train bias weights, set to 'none' for attention layers
            task_type='SEQ_CLS'  # specifying the task type as sequence classification
        )
        return lora_config

    def get_model_config(self, model, tokenizer):
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, self.lora)
        model.config.pad_token_id = tokenizer.pad_token_id

