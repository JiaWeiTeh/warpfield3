#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 15:21:52 2023

@author: Jia Wei Teh
"""

# get parameters
selected_language = 'en'

prompt_raw = {
    
    'Hello':{ 
            'cn': '哈啰',
            'fr': 'Bonjour',
            },
    
    'Welcome to':{
        'cn':  '欢迎使用',
        },
    
    '[Version 3.0] 2022. All rights reserved.':{
        'cn': '[版本 3.0] 2022年。版权所有。',
        },
    
    }

from random import randrange

prompt = {}

if selected_language == 'en':
    for key, val in prompt_raw.items():
        prompt[key] = key
elif selected_language == 'katze':
    for key, val in prompt_raw.items():
        meow_length = ''
        while True:
            meow_length += 'Meo'+'o'*randrange(5)+'w' 
            if randrange(2) == 1:
                break
            meow_length += ' '
            
        prompt[key] = meow_length + ['?', '.', '!','.','??'][randrange(5)]
else:
    for key, val in prompt_raw.items():
        try:
            prompt[key] = val[selected_language]
        except: 
            prompt[key] = key




