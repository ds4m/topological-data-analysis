# -*- coding: utf-8 -*-
"""
Functions to produce, manipulate, and compare Euler characteristic curves

@author: Elchanan Solomon
"""

import numpy as np
import matplotlib.pyplot as plt

def build_complexes(image,cutoff):
    """
    Builds a cubical complex from the greyscale image
    
    Parameters
    ----------
    image - m x n greyscale image
    cutoff - cutoff value for converting greyscale image to binary image
    
    Returns
    ----------
    A tuple of arrays (vertices,edges,squares)
    vertices -- an array of locations in the grid
    edges -- an array of edges (pairs of locations) in the grid
    squares -- an array of squares (quadruples of locations) in the grid
    """
    #We instantiate empty arrays
    vertices = []
    edges = []
    squares = []
    #We record the shape of the data
    (m,n) = image.shape
    #We convert the greyscale image into a binary image, using the cutoff value
    binary_image = np.where(image > cutoff, 1,0)

    #We iterate over locations in the grid    
    for i in range(m):
        for j in range(n):
            #If our location has binary value '1', we add it to our vertex array
            if binary_image[i,j]:
                vertices.append((i,j))            
            #If our location is not in the last column or row, we check if it the location
            #to the right or bottom is also '1'. If so, we add these as edges
            if i<(m-1) and j<(n-1):
                if binary_image[i,j] and binary_image[i,j+1]:
                    edges.append(((i,j),(i,j+1)))
                if binary_image[i,j] and binary_image[i+1,j]:
                        edges.append(((i,j),(i+1,j)))
                        
            #We also check if our location is the top-left corner of a square of '1's,
            #in which case we store that square in our square array
                if binary_image[i,j] and binary_image[i,j+1] and binary_image[i+1,j] and binary_image[i+1,j+1]:
                    squares.append(((i,j),(i,j+1),(i+1,j),(i+1,j+1)))
            #Moving down the last column, we check for vertical edges
            if i<(m-1) and j==(n-1):
                if binary_image[i,j] and binary_image[i+1,j]:
                    edges.append(((i,j),(i+1,j)))  
            #Moving down the last row, we check for horizontal edges        
            if i==(m-1) and j<(n-1):
                if binary_image[i,j] and binary_image[i,j+1]:
                    edges.append(((i,j),(i,j+1)))                       
                
    return (vertices,edges,squares)

def euler_char(V,E,F):
    """
    Returns the euler characteristic of a triple of arrays as the alternating sum of their lengths.
    
    Parameters
    ---------
    V,E,F -- arrays
    
    Returns
    --------
    Returns the euler characteristic as an integer
    
    """
    return len(V) - len(E) + len(F)

def dot_product(v,u):
    """
    Computes the dot product of two vectors
    
    Parameters
    ----------
    v,u - 2-tuples of real values
    
    Returns
    -------
    Returns dot product
    """
    (a,b) = v
    (c,d) = u
    return a*c + b*d

def euler_curve(vertices,edges,squares,start,stop,step,direction):
    """
    Returns the euler characteristic curve of simplicial complex using a directional sweeping filtration.
    
    Parameters
    ----------
    vertices -- an array of locations in the grid
    edges -- an array of edges (pairs of locations) in the grid
    squares -- an array of squares (quadruples of locations) in the grid
    start -- lower value for filtration
    stop -- upper value for filtration
    step -- discrete step taken in each iteration
    direction -- vector to take dot product with, stored as 2-tuple
    
    Returns
    -------
    s_vals -- The height values used in the filtration
    e_curve -- An array of euler characteristic values
    
    """
    #instatiate the height variable
    s = start
    #prepare the array of euler characteristic values
    e_curve = []
    #prepare the array of height_values
    s_vals = []
    #Make a deep copy of the simplicial complex arrays. These arrays will list
    #the simplices not yet in our sublevel set
    V_out = vertices[:]
    E_out = edges[:]
    F_out = squares[:]
    #Empty arrays storing simplices already included in our sublevel set
    V_in = []
    E_in = []
    F_in = []
    #Iterate until max height is reached
    while s<= stop:
        #For every verex not yet included
        for v in V_out[:]:
            #Check if it should be included in the sublevel set, if so...
            if dot_product(v,direction) <=(s+0.0001):
                #Add it to V_in and remove it from V_out
                V_in.append(v)
                V_out.remove(v)
        #Now, check all the edges not yet included. For each edge...     
        for e in E_out[:]:
            #find its constituent vertices
            (v_1,v_2) = e
            #and check if they are in V_in, if so...
            if v_1 in V_in and v_2 in V_in:
                #add the edge to E_in and remove it from E_out
                E_in.append(e)
                E_out.remove(e)
        #Now, check all the faces not yet included. For each face...        
        for f in F_out[:]:
            #find its constituent vertices
            (v_1,v_2,v_3,v_4) = f
            #and check if they are in V_in, if so...
            if v_1 in V_in and v_2 in V_in and v_3 in V_in and v_4 in V_in:
                #add the face to F_in and remove it from F_out
                F_in.append(f)
                F_out.remove(f)
        #Now, compute the euler characteristic of the present subcomplex, and
        #add it to e_curve        
        e_curve.append(euler_char(V_in,E_in,F_in))
        #Also, store the present height value
        s_vals.append(s)
        #Increment the height and repeat...
        s += step
    return (s_vals,e_curve) 

def array_of_smooth_euler_curves(vertices,edges,squares,start,stop,step,directions):
    """
    Computes an array of smoothed euler characteristic curves along a set of directions

    Parameters
    ----------
    vertices -- an array of locations in the grid
    edges -- an array of edges (pairs of locations) in the grid
    squares -- an array of squares (quadruples of locations) in the grid
    start -- lower value for filtration
    stop -- upper value for filtration
    step -- discrete step taken in each iteration
    directions -- set of vectors to take dot product with, stored as 2-tuples   
    
    Returns
    ---------
    ec_array -- an array of Euler characteristic curves
    
    """
    #Initialize the array of ec curves
    ec_array = []
    #For each direction...
    for direc in directions:
        #Compute its ec curves and add it to the array
        ec_array.append(smooth_curve(euler_curve(vertices,edges,squares,start,stop,step,direc)[1]))
    return ec_array  
            
def display_subset(image,direction,height):
    """
    Plots an subcomplex, given a direction and a height.
    
    Parameters
    ----------
    image - the m x n binary image
    direction - a two-tuple vector
    height - a real value to determine the sublevel set
    """
    #Make an empty mxn array
    subslice = np.zeros(image.shape)
    (m,n) = subslice.shape
    # For each location in this array...
    for i in range(m):
        for j in range(n):
            #Give it a value of '1' if it sits in the sublevel set
            if dot_product((i,j),direction) <= height + 0.0001:
                subslice[i,j] = 1
    #Intersect this half-plane with our image using logical and            
    sub_image = np.logical_and(image,subslice)
    #plot this intersection, which gives the sublevel set
    plt.imshow(sub_image, cmap='gray_r')
    plt.show()
    
def smooth_curve(curve):
    """
    Takes an discrete curve and smooths it
    
    Parameters
    ----------
    vals -- array of x-values
    curve -- array of function values
    
    Returns
    -------
    smoothed_curve -- an array of the same length as curve    

    """
    #Find the number of elements in our array
    n = len(curve)
    #and invert that to get step, which is like a "delta x"
    step = 1/(n+0.0)
    #instatiate the smooth curve
    smoothed_curve = []
    #Iterating through our curve...
    for i in range(n):
        # add the initial curve value, multiplied by step
        if i==0:
            smoothed_curve.append(curve[0]*step)
        #continue to take a cumulative sum, adding the prior value of smoothed_curve to the
        #present value of curve, multiplied by step
        else:
            smoothed_curve.append(smoothed_curve[i-1] + curve[i]*step)
    return smoothed_curve
   

def l2dist(f,g):
    """
    Computes the normalized L2 distance between 2 functions
    
    Parameters
    ----------
    f,g -- functions stored as arrays
    
    Returns
    -------
    sum -- square root of Riemann sum for (f-g)^2
    
    """
    #step will be the "delta x" in our Riemann sum
    n = len(f)
    step = 1/(n+0.0)
    sum = 0
    # Compute the Riemann sum of (f-g)^2
    for i in range(n):
        sum+= step*((f[i]-g[i])**2)
    #Return its square root    
    return np.sqrt(sum)

def l2_array_dist(A,B):
    n = len(A)
    sum = 0.0
    for i in range(n):
        sum += l2dist(A[i],B[i])
    return sum/n

def image_dist(im1,im2,directions = [(1,0),(0,1),(-1,0),(0,-1)],cutoff=20):
    """
    Computes the normalized L2 image distance between two images along a set of directions
    
    Parameters
    ----------
    im1,im2 -- greyscale images
    directions -- array of 2-tupes, default is [(1,0),(0,1),(-1,0),(0,-1)]
    cutoff -- cutoff value for simplicial complex, default is 20
    """
    sum=0
    #Build simplicial complexes for our images
    (vertices1,edges1,squares1) = build_complexes(im1,20)
    (vertices2,edges2,squares2) = build_complexes(im2,20)
    #For each direction, compute euler characteristic curve
    for direc in directions:
        (vals1,curve1) = smooth_curve(euler_curve(vertices1,edges1,squares1,0,50,5,direc))
        (vals2,curve2) = smooth_curve(euler_curve(vertices2,edges2,squares2,0,50,5,direc))
        #compute the l2 distance between the curves, and add it to our running sum
        sum += l2dist(curve1,curve2)
    #Return the sum of distances, normalized by the number of directions
    return sum/len(directions)


def complex_dist(com1,com2,directions = [(1,0),(0,1),(-1,0),(0,-1)]):
    """
    Like image distance, but the complexes are given in advance
    
    Parameters
    ----------
    im1,im2 -- greyscale images
    directions -- array of 2-tupes, default is [(1,0),(0,1),(-1,0),(0,-1)]
    """
    sum=0
    (vertices1,edges1,squares1) = com1
    (vertices2,edges2,squares2) = com2
    #For each direction, compute euler characteristic curve
    for direc in directions:
        (vals1,curve1) = smooth_curve(euler_curve(vertices1,edges1,squares1,0,50,5,direc))
        (vals2,curve2) = smooth_curve(euler_curve(vertices2,edges2,squares2,0,50,5,direc))
        #compute the l2 distance between the curves, and add it to our running sum
        sum += l2dist(curve1,curve2)
    #Return the sum of distances, normalized by the number of directions
    return sum/len(directions)


def most_common(lst):
    return max(set(lst), key=lst.count)


def predict_labels(train_data,train_labels,test,directions,knn=1):
    """"
    Predicts labels for test data based on training data,labels, using K-nearest-neighbours
    
    Parameters
    ----------
    train_data -- training data set
    train_data_labels -- training labels
    test -- testing data
    directions -- set of directions to compute EC curves for
    knn -- number of nearest neighbours to consult
    
    Returns
    ---------
    labels -- array of predicted labels
    
    """
    #Initiate array to store, for each test image, its array of EC curves
    test_ec_arrays = []
    #For each test image...
    for fig in test:
        #Build its complexes
        (vertices,edges,squares) = build_complexes(fig,20)
        #And compute its set of EC curves
        test_ec_arrays.append(array_of_smooth_euler_curves(vertices,edges,squares,-100,100,1,directions))
    #Initiate array to store, for each training image, its array of EC curves
    train_ec_arrays = []
    #For each training image...
    for image in train_data:
        #Build its complexes
        (vertices,edges,squares) = build_complexes(image,20)
        #And compute its EC_curves
        train_ec_arrays.append(array_of_smooth_euler_curves(vertices,edges,squares,-100,100,1,directions))
    #Build a matrix of zeros to store distances between testing and training images
    (a,b) = (len(test),len(train_data))
    dist_matrix = np.zeros((a,b))
    #For each pair of training and testing images
    for i in range (a):
        for j in range(b):
            #Compute their distance using EC curves
            dist_matrix[i,j] = l2_array_dist(test_ec_arrays[i],train_ec_arrays[j])
    #Initiate array of labels        
    labels = []
    #For each test image...
    for i in range(a):
        #Sort the row of training labels by corresponding distance of the training image to the test image        
        sorted_labels = [x for _,x in sorted(zip(dist_matrix[i,:],train_labels))]
        #Pick the most common label among the k nearest neighbours
        labels.append(most_common(sorted_labels[:knn]))
    return labels

