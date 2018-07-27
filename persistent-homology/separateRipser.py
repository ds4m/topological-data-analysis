#!/bin/env python 

# This function takes a ripser text file and creates 2 new text files containing the barcode intervals for a single dimension
# in a format that is compatible with hera. 
#
#
# Melissa R. McGuirl

import sys
import numpy as np
import glob
import argparse 


def main():

    # Set up arguments and help
    descriptor = "Separates ripser output by dimension, output can be used as input for hera to compute distances"
    parser = argparse.ArgumentParser(description = descriptor)
    parser.add_argument('-i', '--indir',
                        required = True,
                        action = 'store',
                        default = './',
                        help = 'provide the path to folder containing the Ripser files')
    parser.add_argument('-o', '--outdir',
                        required = True,
                        action = 'store',
                        help = 'provide the name/path of desired output folder')


    args = parser.parse_args()
    DIR = args.indir
    OUT = args.outdir
    
    file_list = glob.glob(DIR + '/*.txt')
    # for each ripser file, separate bars by dimension. Print to new files.
    for file in file_list:

        inFile = open(file, 'r')
        lines = inFile.read().splitlines()
        outFile0 = OUT + '/dim0bars_' + file[len(DIR)::]
        outFile1 = OUT + '/dim1bars_' + file[len(DIR)::]
        outFile0 = open(outFile0, 'w')
        outFile1 = open(outFile1, 'w')

  
        count = 0
        for line in lines: 
            # first line of ripser output
            if count ==0:
                line = line.split(',')
                endpoint = line[1]
                count = 1 
            elif count ==1:
                count = 2
            elif count ==2:

                if line != '' and line != '0, ':
                    line = line.split(',')
                    outFile0.write('%s %s \n'%(line[0], line[1]))
                
                elif line == '0, ':
                    outFile0.write('%s %s \n'%('0', endpoint))
                    count = 3
                else:
                    count = 3
            elif count ==3:
                count = 4
            elif count == 4:
                if line != '':
                    line = line.split(',')
                    outFile1.write('%s %s \n'%(line[0], line[1]))
                else:
                    count = 5


    #Add (0,0) in case there are no bars.

        outFile0.write('0 0 \n')
        outFile1.write('0 0 \n')


        inFile.close()
        outFile0.close()
        outFile1.close()


if __name__ == "__main__":
    
    
    
    main()
