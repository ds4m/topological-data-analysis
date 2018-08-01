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
    # Sample vectors from a normal distribution, ie from a Gaussian distribution, in Euclidean space.
    vectors = np.random.randn(ndim, npoints)
    # Normalize each vector to have length one. These are our uniformly distributed points on the sphere.
    vectors /= np.linalg.norm(vectors, axis=0)
    np.savetxt('point_clouds/sphere_points.txt', np.transpose(vectors), delimiter=', ')

if __name__ == "__main__":
    
    main()







	
			

