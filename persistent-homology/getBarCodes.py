#!/bin/env python

'''
This is a function that processes a collection of distance matrices and runs 
the data through ripser to get the corresponding barcode diagrams. 
The barcodes are then saved to a text file.

Ripser is available freely at https://github.com/Ripser/ripser

Author: Melissa McGuirl
'''
import numpy as np
import sys
import os
import glob
import os.path
import argparse 


def main():
    # Set up arguments and help
    descriptor = "Processes a collection of distance matrices and saves the corresponding barcodes to a text file. Ripser must be in path."
    parser = argparse.ArgumentParser(description = descriptor)

    parser.add_argument('-i', '--indir',
                        required = True,
                        action = 'store',
                        default = './',
                        help = 'provide the path to folder containing distance matrices. We assume all files in folder are distance mats which are compatible with Ripser.')
    parser.add_argument('-o', '--outdir',
                        required = True,
                        action = 'store',
                        help = 'provide the name/path of desired output folder for barcode data')

    args = parser.parse_args()
    DIR = args.indir
    OUT = args.outdir
    # Get list of all files in distance mat directory 
    files = glob.glob(DIR + '/*.txt')
    #For each file in distance mat directory, run ripser. 
    for file in files:
        #Run ripser on each barcode file in the current folder corresponding to black stripes, save output to RipserFile
         RipserFile = OUT + '/ripser_' + file[len(DIR) ::]
         #if os.path.isfile(RipserFile) == False:
         cmd = "./ripser %s | cut -f 2 -d: | awk 'NR != 2 {print}' | cut -c 3- | sed 's/.$//' > %s" % (file, RipserFile)
         os.system(cmd)


if __name__ == "__main__":
    
       
    main()


        
        


