#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 13:37:22 2023

@author: Jia Wei Teh
"""
import time
import os
import sys
from src.warpfield.functions.dictionary import prompt
# get parameter
from src.input_tools import get_param
warpfield_params = get_param.get_param()

def display():
    
    # display logo for WARPFIELD
    show_logo()
    print('\t\t      --------------------------------------------------')
    print('\t\t      '+prompt['Welcome to']+' \033[32m'+link('https://github.com/JiaWeiTeh/warpfield3', 'WARPFIELD')+'\033[39m!\n')
    print('\t\t      Notes:')
    print('\t\t         - Documentation can be found \033[32m'+link('https://warpfield3.readthedocs.io/en/latest/', 'here')+'\033[39m.')
    print('\t\t         - \033[1m\033[96mBolded text\033[0m highlights the designated')
    print('\t\t           locations of saved files.')
    print('\t\t         - \033[1m\033[94mThis is warning/info\033[0m but nothing is huge.\n')
    print('\t\t      '+prompt['[Version 3.0] 2022. All rights reserved.'])
    print('\t\t      --------------------------------------------------')
    # show initial parameters
    show_param()
    
    return


def show_logo():
    
    print(r"""
          ,          __     __   ______   ______   ______  ______  __   ______   __       _____    
       \  :  /      /\ \  _ \ \ /\  __ \ /\  == \ /\  == \/\  ___\/\ \ /\  ___\ /\ \     /\  __-.  
    `. __/ \__ .'   \ \ \/ ".\ \\ \  __ \\ \  __< \ \  _-/\ \  __\\ \ \\ \  __\ \ \ \____\ \ \/\ \ 
    _ _\     /_ _    \ \__/".~\_\\ \_\ \_\\ \_\ \_\\ \_\   \ \_\   \ \_\\ \_____\\ \_____\\ \____- 
       /_   _\        \/_/   \/_/ \/_/\/_/ \/_/ /_/ \/_/    \/_/    \/_/ \/_____/ \/_____/ \/____/ 
     .'  \ /  `.      
          '           © J.W. Teh, D. Rahner, E. Pellegrini, et al.                               
        """)

    return 


def link(url, label = None):
    if label is None: 
        label = url
    parameters = ''
    # OSC 8 ; params ; URL ST <name> OSC 8 ;; ST 
    escape_mask = '\033]8;{};{}\033\\{}\033]8;;\033\\'

    return escape_mask.format(parameters, url, label)

def show_param():
    # print some useful information
    print("Loading parameters:")
    print(f'\tmodel name: {warpfield_params.model_name}')
    print(f'\tlog_mCloud: {warpfield_params.log_mCloud}')
    print(f'\tSFE: {warpfield_params.sfe}')
    print(f'\tmetallicity: {warpfield_params.metallicity}')
    print(f'\tdensity profile: {warpfield_params.dens_profile}')
    # shorten
    relpath = os.path.relpath(warpfield_params.out_dir, os.getcwd())
    print(f"\033[1m\033[96mSummary: {relpath}/{warpfield_params.model_name}{'_summary.txt'}\033[0m")
    filename =  relpath + '/' + warpfield_params.model_name+ '_config.yaml'
    print(f'\033[1m\033[96mVerbose yaml: {filename}\033[0m')

    return





