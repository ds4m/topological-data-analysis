#!/bin/env python 
#
# A script to take a transformed Ripser output file and create persistence diagrams
#
# Author: Melissa R McGuirl
 
import sys
import numpy as np
import argparse 
import matplotlib.pyplot as plt
import glob
def main():
	descriptor = "plots persistent diagrams for a collection of intervals in dimension 0 and 1"
	parser = argparse.ArgumentParser(description = descriptor)
	parser.add_argument('-i', '--indir',
                        required = True,
                        action = 'store',
                        default = './',
                        help = 'provide the path to folder containing the barcode files')
	parser.add_argument('-o', '--outdir',
                        required = True,
                        action = 'store',
                        help = 'provide the name/path of desired output folder to store images')



	args = parser.parse_args()
	DIR = args.indir
	OUT = args.outdir

	files = glob.glob(DIR + '/*.txt')
	for file in files:
		inFile = open(file, 'r')
		lines = inFile.read().splitlines()
		x = []
		y = []
		for i in xrange(len(lines)):
			coords = lines[i].split(' ')
			x.append(float(coords[0]))
			y.append(min(800, float(coords[1])))

		
		fig = plt.figure()
		plt.scatter(x, y)
		plt.xlabel('birth')		
		plt.ylabel('death')
		# User should modify this part to get appropriate titles/savenames 
		fig.suptitle(file[len(DIR) :-4])
		fig.savefig(OUT + '/' + file[len(DIR) :-4])
                plt.close(fig)
		
if __name__ == "__main__":
    
    
	
	main()






	
			

