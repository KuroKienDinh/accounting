#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    model.py
# @Author:      Kuro
# @Time:        7/16/2024 10:48 AM
import torch
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from torch import Tensor, nn
from transformers import AutoModel, BitsAndBytesConfig, AutoModelForSequenceClassification


class BaseModel:
    def __init__(self, pretrained_model, device='cuda'):
        self.model = AutoModel.from_pretrained(pretrained_model, trust_remote_code=True).to(device)
        self.device = device

    def get_model(self):
        return self.model

    @staticmethod
    def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]  # Return last token
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class GemmaModel(BaseModel):
    def __init__(self, pretrained_model, device='cuda', n_classes=2):
        super().__init__(pretrained_model, device=device)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.linear_text = nn.Linear(768, 512)  # Assuming BERT-like output size of 768
        self.output = nn.Linear(512, n_classes)

    def forward(self, input_ids: Tensor, attention_mask: Tensor):
        last_hidden_states = self.model(input_ids, attention_mask=attention_mask).last_hidden_state
        pooled_output = self.last_token_pool(last_hidden_states, attention_mask)
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.linear_text(pooled_output)
        pooled_output = self.relu(pooled_output)
        return self.output(pooled_output)


class MistralModel(BaseModel):
    def __init__(self, pretrained_model, tokenizer, device='cuda', num_labels=3):
        super().__init__(pretrained_model, device)
        self.num_labels = num_labels
        self.device = device
        self.pretrained_model = pretrained_model
        self.tokenizer = tokenizer

    def model_train(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,  # enable 4-bit quantization
            bnb_4bit_quant_type='nf4',  # information theoretically optimal dtype for normally distributed weights
            bnb_4bit_use_double_quant=True,  # quantize quantized weights
            bnb_4bit_compute_dtype=torch.bfloat16  # optimized fp format for ML
        )

        lora_config = LoraConfig(
            r=16,  # the dimension of the low-rank matrices
            lora_alpha=8,  # scaling factor for LoRA activations vs pre-trained weight activations
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            lora_dropout=0.05,  # dropout probability of the LoRA layers
            bias='none',  # whether to train bias weights, set to 'none' for attention layers
            task_type='SEQ_CLS'  # specifying the task type as sequence classification
        )

        model = AutoModelForSequenceClassification.from_pretrained(self.pretrained_model, quantization_config=quantization_config,
                                                                   num_labels=self.num_labels).to(self.device)

        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.config.pad_token_id = self.tokenizer.pad_token_id
        return model

    def model_test(self):
        model = AutoModelForSequenceClassification.from_pretrained(self.pretrained_model, num_labels=self.num_labels).to(self.device)
        model.config.pad_token_id = self.tokenizer.pad_token_id
        model.eval()
        return model
