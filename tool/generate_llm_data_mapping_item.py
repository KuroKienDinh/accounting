#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    generate_llm_data_mapping_item.py
# @Author:      Kuro
# @Time:        2/4/2025 10:46 AM
import json

import pandas as pd

# Specify the path to your CSV file
file_path = "../dataset/10180_item_category.csv"


# Split the header line by comma
header = ["global_category", "item_name", "phi_category", "phi_subcategory"]

# Now read the rest of the file using semicolon as the delimiter.
# We set header=None because we're providing our own header names.
df = pd.read_csv(file_path, sep=";", header=None, names=header)


# df = df[["item_name", "global_category"]]
df.dropna(inplace=True)


# system_msg = """
# You are an advanced AI language model with specialized expertise in accounting classifications. Your task is to map a provided \"item_name\" to its corresponding global accounting details. For each input, you must generate a JSON object containing the following keys:
#
# - \"item_name\": The original item name provided by the user.
# - \"global_account_name\": The overarching account classification.
# - \"category_1\": The primary accounting category.
# - \"category_2\": The subcategory corresponding to the primary category.
#
# Instructions:
# 1. Input: A single \"item_name\" string.
# 2. Output: A JSON object strictly following this structure:
#    {
#      \"item_name\": \"<original item name>\",
#      \"global_account_name\": \"<determined global account name>\",
#      \"category_1\": \"<determined primary category>\",
#      \"category_2\": \"<determined subcategory>\"
#    }
# 3. Your output must be in valid JSON format with no additional text, explanations, or commentary.
# 4. If you are uncertain about the correct classification for any field, assign the value \"Unknown\" to that field.
# 5. Leverage your comprehensive accounting knowledge to ensure that your mappings are as accurate and professional as possible.
#
# Example:
# Input: \"Support Januar\"
# Output:
# {
#   \"global_account_name\": \"Operating Expenses\",
#   \"category_1\": \"Licenses & Permits\",
#   \"category_2\": \"Software Licenses\"
# }
# """


system_msg = """
You are an advanced AI language model with specialized expertise in accounting classifications. Your task is to map a provided "item_name" to its corresponding global accounting details and to evaluate your confidence in the classification. For each input, generate a JSON object containing the following keys:

- "item_name": The original item name provided by the user.
- "global_account_name": The overarching account classification.
- "confidence_status": Your confidence level in the mapping global account name; set this to "Sure" if your confidence is above 95% and "Not Sure" if it is below 95%.
- "category_1": The primary accounting category.
- "category_2": The subcategory corresponding to the primary category.

Instructions:
1. Input: A single "item_name" string.
2. Output: A JSON object strictly following this structure:
   {
     "global_account_name": "<determined global account name>",
     "confidence_status": "<Sure or Not Sure>",
     "category_1": "<determined primary category>",
     "category_2": "<determined subcategory>"
   }
3. Your output must be in valid JSON format with no additional text, explanations, or commentary.
4. If you are uncertain about the correct classification for any field, assign the value "Unknown" to that field.
5. Use your comprehensive accounting knowledge to ensure that your mappings and confidence assessments are as accurate and professional as possible.
"""

with open("../dataset/llm_data_mapping_item_promp_2.txt", "w", encoding="utf-8") as json_output:
    for i, row in df.iterrows():
        output = {
            "global_account_name": row["global_category"],
            "confidence_status": "Sure",
            "category_1": row["phi_category"],
            "category_2": row["phi_subcategory"]
        }
        record = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": row["item_name"]},
            {"role": "assistant", "content": json.dumps(output, ensure_ascii=False)}
        ]
        json_record = json.dumps(record, ensure_ascii=False)
        json_output.write(json_record + "\n")

