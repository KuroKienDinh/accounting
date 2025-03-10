#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    generate_llm_data_mapping_item.py
# @Author:      Kuro
# @Time:        2/4/2025 10:46 AM
import json

import pandas as pd


# file_path = "../dataset/10180_item_category.csv"
file_path = "../dataset/train_10180_final.csv"
df = pd.read_csv(file_path)
df.dropna(inplace=True, how='all')
df.reset_index(drop=True, inplace=True)

system_msg = """
You are an advanced AI language model with specialized expertise in accounting classifications. Your task is to map a provided "item_name" to its corresponding global accounting details and to evaluate your confidence in the classification. For each input, generate a JSON object containing the following keys:

- "item_name": The original item name provided by the user.
- "global_account_name": The overarching account classification.
- "confidence_status": Your confidence level in the mapping global account name; set this to "Sure" if your confidence is above 95% and "Not Sure" if it is below 95%.

Instructions:
1. Input: A single "item_name" string.
2. Output: A JSON object strictly following this structure:
   {
     "global_account_name": "<determined global account name>",
     "confidence_status": "<Sure or Not Sure>",
   }
3. Your output must be in valid JSON format with no additional text, explanations, or commentary.
4. If you are uncertain about the correct classification for any field, assign the value "Unknown" to that field.
5. Use your comprehensive accounting knowledge to ensure that your mappings and confidence assessments are as accurate and professional as possible.
"""

with open("../dataset/10180_data_mapping_item_account.txt", "w", encoding="utf-8") as json_output:
    for i, row in df.iterrows():
        output = {
            "global_account_name": row["system_account_name"],
            "confidence_status": "Sure"
        }
        record = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": row["item_name"]},
            {"role": "assistant", "content": json.dumps(output, ensure_ascii=False)}
        ]
        json_record = json.dumps(record, ensure_ascii=False)
        json_output.write(json_record + "\n")




