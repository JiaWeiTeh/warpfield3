#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 15:39:35 2023

@author: Jia Wei Teh

This script contains functions that handle printing information in the terminal.
"""


def phase1():
    
    print('\n\t\t      --------------------------------------------------')
    print('\t\t      Entering phase 1 (energy-driven phase)')
    print('\t\t      --------------------------------------------------')
    
    return


def phase0():
    
    print('\n\t\t      --------------------------------------------------')
    print('\t\t      Initialising bubble')
    print('\t\t      --------------------------------------------------')
    
    return





class cprint:
    # A class that deals with printing with colours in terminal. 
    # e.g., print(f'{cprint.BOLD}This text is bolded{cprint.END}  but this isnt.')
    
    # bolded text to signal that a file is being saved
    symbol = '\u27B3 '
    BOLD = symbol+'\033[1m\033[96m'
    # aliases
    SAVE = BOLD
    FILE = BOLD
    
    # Link
    LINK = '\033[32m'
    
    # Warning message, but code runs still. 
    WARN = '\033[1m\033[94m'
    
    # Blink
    BLINK = '\033[5m'
    
    # FAIL
    FAIL = '\033[1m\033[91m'
    
    # END and clear all colours. This should be included in the end of every operations.
    END = '\033[0m'


