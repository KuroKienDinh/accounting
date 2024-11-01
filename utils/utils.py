#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    utils.py
# @Author:      Kuro
# @Time:        7/23/2024 3:23 PM
import os


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

