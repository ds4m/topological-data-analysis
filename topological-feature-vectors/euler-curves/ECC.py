# -*- coding: utf-8 -*-
"""
Functions to produce, manipulate, and compare Euler characteristic curves

@author: Elchanan Solomon
"""

import numpy as np
import matplotlib.pyplot as plt
from progressbar import *
widgets = ['Run: ', Percentage(), ' ', Bar(marker=RotatingMarker()),
               ' ', ETA(), ' ', FileTransferSpeed()]

def build_complexes(image,cutoff):
    """
    Builds a cubical complex from the greyscale image
    
    Parameters
    ----------
    image - m x n greyscale image
    cutoff - cutoff value for converting greyscale image to binary image
    
    Returns
    ----------
    A tuple of lists (vertices,edges,squares)
    vertices -- an list of locations in the grid
    edges -- an list of edges (pairs of locations) in the grid
    squares -- an list of squares (quadruples of locations) in the grid
    """
    #We instantiate empty lists
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
            #If our location has binary value '1', we add it to our vertex list
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
            #in which case we store that square in our square list
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
    Returns the euler characteristic of a triple of lists as the alternating sum of their lengths.
    
    Parameters
    ---------
    V,E,F -- lists
    
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
    vertices -- an list of locations in the grid
    edges -- an list of edges (pairs of locations) in the grid
    squares -- an list of squares (quadruples of locations) in the grid
    start -- lower value for filtration
    stop -- upper value for filtration
    step -- discrete step taken in each iteration
    direction -- vector to take dot product with, stored as 2-tuple
    
    Returns
    -------
    s_vals -- The height values used in the filtration
    e_curve -- An list of euler characteristic values
    
    """
    #instatiate the height variable
    s = start
    #prepare the list of euler characteristic values
    e_curve = []
    #prepare the list of height_values
    s_vals = []
    #Make a deep copy of the simplicial complex lists. These lists will list
    #the simplices not yet in our sublevel set
    V_out = vertices[:]
    E_out = edges[:]
    F_out = squares[:]
    #Empty lists storing simplices already included in our sublevel set
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

def list_of_smooth_euler_curves(vertices,edges,squares,start,stop,step,directions):
    """
    Computes an list of smoothed euler characteristic curves along a set of directions

    Parameters
    ----------
    vertices -- an list of locations in the grid
    edges -- an list of edges (pairs of locations) in the grid
    squares -- an list of squares (quadruples of locations) in the grid
    start -- lower value for filtration
    stop -- upper value for filtration
    step -- discrete step taken in each iteration
    directions -- set of vectors to take dot product with, stored as 2-tuples   
    
    Returns
    ---------
    ec_list -- an list of Euler characteristic curves
    
    """
    #Initialize the list of ec curves
    ec_list = []
    #For each direction...
    for direc in directions:
        #Compute its ec curves and add it to the list
        ec_list.append(smooth(euler_curve(vertices,edges,squares,start,stop,step,direc)[1],10))
    return ec_list  
            
def display_subset(image,direction,height):
    """
    Plots an subcomplex, given a direction and a height.
    
    Parameters
    ----------
    image - the m x n binary image
    direction - a two-tuple vector
    height - a real value to determine the sublevel set
    """
    #Make an empty mxn list
    subslice = np.zeros(image.shape)
    (m,n) = subslice.shape
    # For each location in this list...
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


    

def l2dist(f,g):
    """
    Computes the normalized L2 distance between 2 functions
    
    Parameters
    ----------
    f,g -- functions stored as lists
    
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

def l2_list_dist(A,B):
    """
    Computes the normalized L2 distance between 2 equal-sized lists of functions
    
    Parameters
    ----------
    A,B -- lists of functions
    
    Returns
    -------
    sum -- normalized sum of L^2 distances between functions in the lists
    
    """
    #Compute the number of functions in our lists
    n = len(A)
    sum = 0.0
    #for each index...
    for i in range(n):
        #add up the L2 distance between the corresponding functions in each list
        sum += l2dist(A[i],B[i])
    #return the sum divided by the total number of functions    
    return sum/n

def image_dist(im1,im2,directions = [(1,0),(0,1),(-1,0),(0,-1)],cutoff=20):
    """
    Computes the normalized L2 image distance between two images along a set of directions
    
    Parameters
    ----------
    im1,im2 -- greyscale images
    directions -- list of 2-tupes, default is [(1,0),(0,1),(-1,0),(0,-1)]
    cutoff -- cutoff value for simplicial complex, default is 20
    """
    sum=0
    #Build simplicial complexes for our images
    (vertices1,edges1,squares1) = build_complexes(im1,20)
    (vertices2,edges2,squares2) = build_complexes(im2,20)
    #For each direction, compute euler characteristic curve
    for direc in directions:
        (vals1,curve1) = smooth(euler_curve(vertices1,edges1,squares1,0,50,5,direc),10)
        (vals2,curve2) = smooth(euler_curve(vertices2,edges2,squares2,0,50,5,direc),10)
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
    directions -- list of 2-tupes, default is [(1,0),(0,1),(-1,0),(0,-1)]
    """
    sum=0
    (vertices1,edges1,squares1) = com1
    (vertices2,edges2,squares2) = com2
    #For each direction, compute euler characteristic curve
    for direc in directions:
        (vals1,curve1) = smooth(euler_curve(vertices1,edges1,squares1,0,50,5,direc),10)
        (vals2,curve2) = smooth(euler_curve(vertices2,edges2,squares2,0,50,5,direc),10)
        #compute the l2 distance between the curves, and add it to our running sum
        sum += l2dist(curve1,curve2)
    #Return the sum of distances, normalized by the number of directions
    return sum/len(directions)


def most_common(lst):
    return max(set(lst), key=lst.count)

    
def smooth(y, box_pts):
    """
    Takes an discrete curve and smooths it via convolution
    
    Parameters
    ----------
    y -- the curve
    box_puts -- the width parameter in the convolution
    
    Returns
    -------
    smoothed_curve -- an list of the same length as curve    
    """
    #create the averaging function
    box = np.ones(box_pts)/box_pts
    #convolve curve with the averaging function
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def vectorize(data,directions):
    """
    Turns a list of figures into a list (one per figure) of lists (one per direction) of EC curves
    
    Parameters
    ----------
    data -- list of figures
    directions -- list of 2-tuple directions
    
    Returns
    -------
    test_ec_lists -- list (one per figure) of lists (one per direction) of EC curves
    
    """
    #Initiate list to store, for each image, its list of EC curves
    test_ec_lists = []
    #For each image...
    print("Generating EC Vectors:")
    pbar = ProgressBar(widgets= widgets, maxval=len(data)).start()
    indx = -1    
    for fig in data:
        #Build its complexes
        (vertices,edges,squares) = build_complexes(fig,20)
        #And compute its set of EC curves
        test_ec_lists.append(list_of_smooth_euler_curves(vertices,edges,squares,0,100,1,directions))
        indx+=1
        pbar.update(indx)
    return test_ec_lists    
    


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
    labels -- list of predicted labels
    
    """
    #Initiate list to store, for each test image, its list of EC curves
    test_ec_lists = []
    #For each test image...
    print("Generating EC for Test Set")
    pbar = ProgressBar(widgets= widgets, maxval=len(test)).start()
    indx = -1
    for fig in test:
        #Build its complexes
        (vertices,edges,squares) = build_complexes(fig,20)
        #And compute its set of EC curves
        test_ec_lists.append(list_of_smooth_euler_curves(vertices,edges,squares,0,100,1,directions))
        indx +=1
        pbar.update(indx)
    pbar.finish()
    print("Complete!")
    #Initiate list to store, for each training image, its list of EC curves
    train_ec_lists = []
    #For each training image...
    print("Generating EC for Train Set")
    pbar = ProgressBar(widgets=widgets, maxval=len(train_data)).start()
    indx = -1    
    for image in train_data:       
        #Build its complexes
        (vertices,edges,squares) = build_complexes(image,20)
        #And compute its EC_curves
        train_ec_lists.append(list_of_smooth_euler_curves(vertices,edges,squares,0,100,1,directions))
        indx +=1
        pbar.update(indx)
    pbar.finish()
    print("Complete!")    
    #Build a matrix of zeros to store distances between testing and training images
    (a,b) = (len(test),len(train_data))
    dist_matrix = np.zeros((a,b))
    #For each pair of training and testing images
    print("Computing mutual distances...")
    pbar = ProgressBar(widgets=widgets, maxval=a).start()
    indx = -1
    for i in range (a):
        for j in range(b):
            #Compute their distance using EC curves
            dist_matrix[i,j] = l2_list_dist(test_ec_lists[i],train_ec_lists[j])
        indx+=1
        pbar.update(indx)
    pbar.finish()
    print("Complete!")     
    #Initiate list of labels        
    labels = []
    #For each test image...
    print("Generating labels...")
    pbar = ProgressBar(widgets=widgets, maxval=a).start()
    indx = -1    
    for i in range(a):
        #Sort the row of training labels by corresponding distance of the training image to the test image        
        sorted_labels = [x for _,x in sorted(zip(dist_matrix[i,:],train_labels))]
        #Pick the most common label among the k nearest neighbours
        labels.append(most_common(sorted_labels[:knn]))
        indx+=1
        pbar.update(indx)
    pbar.finish()
    print("Complete!")     
    return labels

