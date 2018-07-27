#!/bin/env python 

'''
A function that takes as input two text files of persistent diagrams in the format 'b d \n' and
computes the bottleneck/Wasserstein distance between them using Hera. Hera is publically 
available at https://bitbucket.org/grey_narn/hera.

Author: Melissa McGuirl 
''' 

import sys, os
import numpy as np
import argparse

# Compute bottleneck dist 
def computeBottleneck(pd1, pd2):
    cmd = 'bottleneck_dist %s %s' % (pd1, pd2)
    d = os.popen(cmd).readlines()
    d = float(d[0])
    return d
# Compute Wasserstein-p distance
def computeWass(pd1, pd2, p):
    cmd = 'wasserstein_dist %s %s %s' % (pd1, pd2, p)
    d =  os.popen(cmd).readlines()
    d = float(d[0])
    return d

def main():
    # Set up help and args
    descriptor = "Computes distance between two persistence diagrams using Hera."
    parser = argparse.ArgumentParser(description = descriptor)
    parser.add_argument('-pd1', '--diagram1', required = True, action = 'store', 
                        help = 'provide path to first barcode')
    parser.add_argument('-pd2', '--diagram2', required = True, action = 'store',
                        help = 'provide path to second barcode')
    parser.add_argument('-m', '--metric', required = True, action = 'store',
                        help = 'specify p for desires p-wasserstein metric, use inf for Bottleneck')

    args = parser.parse_args()
    pd1 = args.diagram1
    pd2 = args.diagram2
    
    p = args.metric
    # call appropriate function 
    if p == 'inf':
        dist = computeBottleneck(pd1, pd2)
    else:
        dist = computeWass(pd1, pd2, p)
    return dist 

if __name__ == "__main__":
    main()
