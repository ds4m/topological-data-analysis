#!/bin/env python 
#
# A script to sample points uniformly at random from the unit sphere
#
# Author: Henry Adams
 
import sys
import numpy as np
import argparse 
import matplotlib.pyplot as plt
def main(npoints=500, ndim=3):
    descriptor = "Sample points uniformly at random from the unit sphere"
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    np.savetxt('point_clouds/sphere_points.txt', np.transpose(vec), delimiter=', ')

if __name__ == "__main__":
    
    main()







	
			

