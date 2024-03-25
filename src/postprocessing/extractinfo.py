#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 16:45:42 2024

@author: Jia Wei Teh

A simple script allowing user to extract certain lines from text files to check the evolution of certain parameters

example: 
    python3 src/postprocessing/extractinfo.py -f testresultv3.txt -s "r2Prime"  -n 1

"""
# library
import argparse
import os

# =============================================================================
# Read in text files
# =============================================================================
# parser
parser = argparse.ArgumentParser()
# Add option to read in file
parser.add_argument('-f', '--file',
                    help = 'Path to your desired .txt file.')
# Add option to read in the line
parser.add_argument('-s', '--string',
                    help = 'Enter the string/line of interest.')
# optional, if want to look at nearby
parser.add_argument('-n', '--nlines',
                    help = 'Number of lines below the requested string.',
                    default = None,
                    nargs = '?',
                    )
# grab argument
args = parser.parse_args()
# open file
file = open(args.file, "r")
# get list of lines in the file, separated by \n
file_lines = file.read().splitlines()
# file size
file_size = os.fstat(file.fileno()).st_size

# go through each line to search for matching string. Can of course
# use regex but not necessary.
# TODO: in the future, allow user to use regex expressions, too.
# record previous line to know evolution
previous_ii = 0

# loop through
for ii, line in enumerate(file_lines):
    if args.string in line: 
        if args.nlines is not None:
            # for cases where the upper/lower boundary exceeds the physical answer
            showlinesabove = (ii - int(args.nlines))
            showlinesbelow = (ii + int(args.nlines))
            
            # option to show above also available, but is now disabled. 
            if showlinesabove < 0:
                showlinesabove = 0
            if showlinesbelow > (file_size-1):
                showlinesbelow = (file_size-1)
            
            # print them in separate rows
            print('\nline:', ii, '[+'+str(ii - previous_ii)+']', file_lines[ii])
            for jj in range(ii+1, showlinesbelow+1):
                print(file_lines[jj])
            
        else:
            print('line:', ii, '[+'+str(ii - previous_ii)+']', line)
            
        previous_ii = ii
# close file.
file.close()






