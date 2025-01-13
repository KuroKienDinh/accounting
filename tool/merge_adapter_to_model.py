#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    merge_adapter_to_model.py
# @Author:      development
# @Time:        12/04/2024 11:41

import argparse
import os
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for processing model data.")
    parser.add_argument("--model_name", type=str, help="Name of the model", default="cdcfahlgren1/natural-functions")
    parser.add_argument("--export_dir", type=str, help="Export directory", default="model_function_export")
    parser.add_argument("--adapters_dir", type=str, help="Adapters directory", required=True)
    args = parser.parse_args()

    base_model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, device_map="cpu")
    model = PeftModel.from_pretrained(base_model, args.adapters_dir)
    model = model.merge_and_unload()

    # Reload tokenizer to save it
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = 'right'

    if not os.path.exists(args.export_dir):
        os.makedirs(args.export_dir)
    model.save_pretrained(args.export_dir)
    tokenizer.save_pretrained(args.export_dir)
