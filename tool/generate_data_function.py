#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    generate_data_function.py
# @Author:      kien.tadinh
# @Time:        4/7/2024 11:41 PM

import random

import pandas as pd
from tqdm import tqdm

with open(("bot_data/user_greeting_en.txt"), "r", encoding='utf-8') as f:
    user_greeting_en = f.read()
    user_greeting_en = eval(user_greeting_en)

with open(("bot_data/user_greeting_de.txt"), "r", encoding='utf-8') as f:
    user_greeting_de = f.read()
    user_greeting_de = eval(user_greeting_de)

with open(("bot_data/bot_greeting_en.txt"), "r", encoding='utf-8') as f:
    bot_greeting_en = f.read()
    bot_greeting_en = eval(bot_greeting_en)

with open(("bot_data/bot_greeting_de.txt"), "r", encoding='utf-8') as f:
    bot_greeting_de = f.read()
    bot_greeting_de = eval(bot_greeting_de)

with open(("bot_data/bot_ask_item_name_en.txt"), "r", encoding='utf-8') as f:
    bot_ask_item_name_en = f.read()
    bot_ask_item_name_en = eval(bot_ask_item_name_en)

with open(("bot_data/bot_ask_item_name_de.txt"), "r", encoding='utf-8') as f:
    bot_ask_item_name_de = f.read()
    bot_ask_item_name_de = eval(bot_ask_item_name_de)

with open(("bot_data/bot_function_en.txt"), "r", encoding='utf-8') as f:
    bot_function_en = f.read()
    bot_function_en = eval(bot_function_en)

with open(("bot_data/bot_function_de.txt"), "r", encoding='utf-8') as f:
    bot_function_de = f.read()
    bot_function_de = eval(bot_function_de)

with open(("bot_data/bot_sorry_en.txt"), "r", encoding='utf-8') as f:
    bot_sorry_en = f.read()
    bot_sorry_en = eval(bot_sorry_en)

with open(("bot_data/bot_sorry_de.txt"), "r", encoding='utf-8') as f:
    bot_sorry_de = f.read()
    bot_sorry_de = eval(bot_sorry_de)

with open(("bot_data/bot_function_quest_en.txt"), "r", encoding='utf-8') as f:
    bot_function_quest_en = f.read()
    bot_function_quest_en = eval(bot_function_quest_en)

with open(("bot_data/bot_function_quest_de.txt"), "r", encoding='utf-8') as f:
    bot_function_quest_de = f.read()
    bot_function_quest_de = eval(bot_function_quest_de)

with open(("bot_data/item_en.txt"), "r", encoding='utf-8') as f:
    item_en = f.read()
    item_en = eval(item_en)

with open(("bot_data/item_de.txt"), "r", encoding='utf-8') as f:
    item_de = f.read()
    item_de = eval(item_de)

with open(("bot_data/user_end_en.txt"), "r", encoding='utf-8') as f:
    user_end_en = f.read()
    user_end_en = eval(user_end_en)

with open(("bot_data/user_end_de.txt"), "r", encoding='utf-8') as f:
    user_end_de = f.read()
    user_end_de = eval(user_end_de)

with open(("bot_data/bot_end_en.txt"), "r", encoding='utf-8') as f:
    bot_end_en = f.read()
    bot_end_en = eval(bot_end_en)

with open(("bot_data/bot_end_de.txt"), "r", encoding='utf-8') as f:
    bot_end_de = f.read()
    bot_end_de = eval(bot_end_de)


class BotData:
    def __init__(self, user_greeting_en, user_greeting_de, bot_greeting_en, bot_greeting_de, bot_ask_item_name_en, bot_ask_item_name_de,
                 bot_function_en, bot_function_de, bot_sorry_en, bot_sorry_de, bot_function_quest_en, bot_function_quest_de, item_en, item_de,
                 user_end_en, user_end_de, bot_end_en, bot_end_de):
        self.system_message = "You are a helpful assistant with access to the following functions. Use them if required -\n{\n    \"name\": \"accounting_invoice\",\n    \"description\": \"Accounting processing with item input\",\n    \"parameters\": {\n        \"type\": \"object\",\n        \"properties\": {\n            \"item_name\": {\n                \"type\": \"object\",\n                \"description\": \"The item name from user's input\"\n            }\n        },\n        \"required\": [\n            \"item_name\"\n        ]\n    }\n}\n"
        self.user_greeting_en = user_greeting_en
        self.user_greeting_de = user_greeting_de
        self.bot_greeting_en = bot_greeting_en
        self.bot_greeting_de = bot_greeting_de
        self.user_end_en = user_end_en
        self.user_end_de = user_end_de
        self.bot_end_en = bot_end_en
        self.bot_end_de = bot_end_de
        self.bot_ask_item_name_en = bot_ask_item_name_en
        self.bot_ask_item_name_de = bot_ask_item_name_de
        self.bot_function_en = bot_function_en
        self.bot_function_de = bot_function_de
        self.bot_sorry_en = bot_sorry_en
        self.bot_sorry_de = bot_sorry_de
        self.bot_function_quest_en = bot_function_quest_en
        self.bot_function_quest_de = bot_function_quest_de
        self.item_en = item_en
        self.item_de = item_de

    def _prompt(self, user_messages, assistant_messages):
        prompt = f"<|im_start|>system\n{self.system_message}<|im_end|>"
        for user_mss, assistant_mss in zip(user_messages, assistant_messages):
            prompt += f"\n<|im_start|>user\n{user_mss}<|im_end|>\n<|im_start|>assistant\n{assistant_mss}<|im_end|>"
        return prompt

    def _greeting(self, lang="en"):
        if lang == "en":
            user_message = random.choice(self.user_greeting_en)
            assistant_message = random.choice(self.bot_greeting_en)
        else:
            user_message = random.choice(self.user_greeting_de)
            assistant_message = random.choice(self.bot_greeting_de)
        return user_message, assistant_message

    def _end_conversation(self, lang="en"):
        if lang == "en":
            user_message = random.choice(self.user_end_en)
            assistant_message = random.choice(self.bot_end_en)
        else:
            user_message = random.choice(self.user_end_de)
            assistant_message = random.choice(self.bot_end_de)
        return user_message, assistant_message

    def _sorry(self, lang="en"):
        if lang == "en":
            user_message = random.choice(self.bot_sorry_en)
            assistant_message = "I'm sorry for any confusion, but as an AI, I am only able to assist with accounting tasks. Could you provide an item name that you need assistance with for accounting?"
        else:
            user_message = random.choice(self.bot_sorry_de)
            assistant_message = "Es tut mir leid für die Verwirrung, aber als KI kann ich nur bei Buchhaltungsaufgaben helfen. Könnten Sie einen Artikelnamen angeben, bei dem Sie Hilfe für die Buchhaltung benötigen?"
        return user_message, assistant_message

    def _ask_item_name(self, lang="en"):
        if lang == "en":
            user_message, assistant_message = random.choice(list(self.bot_ask_item_name_en.items()))
        else:
            user_message, assistant_message = random.choice(list(self.bot_ask_item_name_de.items()))
        return user_message, assistant_message

    # def _bot_function(self, lang="en"):
    #     if lang == "en":
    #         user_message, assistant_message = random.choice(list(self.bot_function_en.items()))
    #     else:
    #         user_message, assistant_message = random.choice(list(self.bot_function_de.items()))
    #     return user_message, assistant_message

    def _bot_function_quest(self, lang="en", all=False):
        _user_messages = []
        _assistant_messages = []
        if lang == "en":
            if not all:
                item = random.choice(self.item_en)
                assistant_message = f"<functioncall> {{\"name\": \"accounting_invoice\", \"arguments\": '{{\"item_name\": \"{item}\"}}'}}"
                user_message = random.choice(self.bot_function_quest_en)
                user_message = user_message.replace("#####", item)
                _user_messages.append(user_message)
                _assistant_messages.append(assistant_message)
                user_message = item
                _user_messages.append(user_message)
                _assistant_messages.append(assistant_message)
            else:
                for item in self.item_en:
                    assistant_message = f"<functioncall> {{\"name\": \"accounting_invoice\", \"arguments\": '{{\"item_name\": \"{item}\"}}'}}"
                    for user_message in self.bot_function_quest_en:
                        user_message = user_message.replace("#####", item)
                        _user_messages.append(user_message)
                        _assistant_messages.append(assistant_message)
                        user_message = item
                        _user_messages.append(user_message)
                        _assistant_messages.append(assistant_message)

        else:
            if not all:
                item = random.choice(self.item_de)
                assistant_message = f"<functioncall> {{\"name\": \"accounting_invoice\", \"arguments\": '{{\"item_name\": \"{item}\"}}'}}"
                user_message = random.choice(self.bot_function_quest_de)
                user_message = user_message.replace("#####", item)
                _user_messages.append(user_message)
                _assistant_messages.append(assistant_message)
                user_message = item
                _user_messages.append(user_message)
                _assistant_messages.append(assistant_message)
            else:
                for item in self.item_de:
                    assistant_message = f"<functioncall> {{\"name\": \"accounting_invoice\", \"arguments\": '{{\"item_name\": \"{item}\"}}'}}"
                    for user_message in self.bot_function_quest_de:
                        user_message = user_message.replace("#####", item)
                        _user_messages.append(user_message)
                        _assistant_messages.append(assistant_message)
                        user_message = item
                        _user_messages.append(user_message)
                        _assistant_messages.append(assistant_message)
        return _user_messages, _assistant_messages

    def conversation_1(self):
        """Greeting"""
        prompts = []
        for lang in ["en", "de"]:
            user_messages = []
            assistant_messages = []
            user_messages_greeting, assistant_messages_greeting = self._greeting(lang)
            user_messages.append(user_messages_greeting)
            assistant_messages.append(assistant_messages_greeting)
            prompt = self._prompt(user_messages, assistant_messages)
            prompts.append(prompt)
        return prompts

    def conversation_2(self):
        """Ask item name"""
        prompts = []
        for lang in ["en", "de"]:
            user_messages = []
            assistant_messages = []
            user_message_ask_item_name, assistant_message_ask_item_name = self._ask_item_name(lang)
            user_messages.append(user_message_ask_item_name)
            assistant_messages.append(assistant_message_ask_item_name)
            prompt = self._prompt(user_messages, assistant_messages)
            prompts.append(prompt)
        return prompts

    def conversation_3(self, all=False):
        """Function item"""
        prompts = []
        for lang in ["en", "de"]:
            user_messages = []
            assistant_messages = []
            # user_message_function, assistant_message_function = self._bot_function(lang, all)
            user_message_functions, assistant_message_functions = self._bot_function_quest(lang, all)
            user_messages.extend(user_message_functions)
            assistant_messages.extend(assistant_message_functions)
            for user_message, assistant_message in zip(user_messages, assistant_messages):
                prompt = self._prompt([user_message], [assistant_message])
                prompts.append(prompt)
        return prompts

    def conversation_4(self):
        """Sorry"""
        prompts = []
        for lang in ["en", "de"]:
            user_messages = []
            assistant_messages = []
            user_message_sorry, assistant_message_sorry = self._sorry(lang)
            user_messages.append(user_message_sorry)
            assistant_messages.append(assistant_message_sorry)
            prompt = self._prompt(user_messages, assistant_messages)
            prompts.append(prompt)
        return prompts

    def conversation_5(self):
        """End conversation"""
        prompts = []
        for lang in ["en", "de"]:
            user_messages = []
            assistant_messages = []
            user_message_end, assistant_message_end = self._end_conversation(lang)
            user_messages.append(user_message_end)
            assistant_messages.append(assistant_message_end)
            prompt = self._prompt(user_messages, assistant_messages)
            prompts.append(prompt)
        return prompts


if __name__ == "__main__":
    args = [user_greeting_en, user_greeting_de, bot_greeting_en, bot_greeting_de, bot_ask_item_name_en, bot_ask_item_name_de,
            bot_function_en, bot_function_de, bot_sorry_en, bot_sorry_de, bot_function_quest_en, bot_function_quest_de, item_en, item_de, user_end_en,
            user_end_de, bot_end_en, bot_end_de]
    bot_data = BotData(*args)
    file_name = "prompt_data_short.xlsx"
    dataset = []
    for i in range(1, 6):  # 6 or 31
        for length_data in tqdm(range(2000)):
            if i == 3:
                prompts = getattr(bot_data, f"conversation_{i}")(all=True)
                dataset.extend(prompts)
                break
            else:
                prompts = getattr(bot_data, f"conversation_{i}")()
                dataset.extend(prompts)
    df = pd.DataFrame(dataset, columns=['data'])
    df.drop_duplicates(subset=['data'], inplace=True)
    df = df.sample(frac=1).reset_index(drop=True)
    df['data'] = df['data'].str.lower()
    df.to_excel(file_name, index=False)
