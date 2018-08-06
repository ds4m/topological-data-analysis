#!/bin/env python 

"""
Cluster comparison in Python

Code taken and modified from: http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html

Additional resources:
    -sklearn user tutorial: http://scikit-learn.org/stable/user_guide.html
    -machine learning with scikit-learn tutorial: http://scikit-learn.org/stable/tutorial/basic/tutorial.html
    -choosing the right estimator: http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

Melissa McGuirl, Brown University. 
Contact: melissa_mcguirl@brown.edu

Date Modified: 07/26/18

As always, we first load in the modules we need to use. 
"""
import time 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn import cluster

def main():
    

    # Create vector of names for  algorithms of interest 
    names = ["K-means, K = 2", "Spectral clustering", "Average linkage clustering"]

    # Make a vector of clustering algorithms that we want to test. Specify input
    # parameters for clustering algorithm. Here we are testing kmeans with k=2,
    # spectral clustering with 2 clusters and nearest-neighbor affinity,
    # and average linkage clustering with 2 clusters and the city block metric,

    cluster_algs = [cluster.KMeans(n_clusters=2),cluster.SpectralClustering(n_clusters = 2,
        affinity="nearest_neighbors"), cluster.AgglomerativeClustering(linkage="average", n_clusters=2,
            affinity="cityblock")]

    # Consider three data sets from Python: data sampled from two noisy moon
    # shapes, data sampled from an inner and outer noisy circle, and noisy
    # blobs. Here we choose a data set of size 1500. 
    n_samples = 1500

    datasets = [make_moons(n_samples=n_samples,noise=0.01, random_state=0),
                make_circles(n_samples=n_samples,noise=0.02, factor=0.5, random_state=1),
                make_blobs(n_samples=n_samples, random_state=8)]

    # Initiate figure for plotting results 
    figure = plt.figure(figsize=(27, 9))
    
    h = 0.2  #Define spatial step size
    i = 1   #Counting variable for plotting 

    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):
        # Preprocess dataset, split into training and test part
        X, y = ds
        X = StandardScaler().fit_transform(X)
        
        # Create a mesh over domain     
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))

        # Plot the three datasets first
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF','#FFA500']) 
        ax = plt.subplot(len(datasets), len(cluster_algs) + 1, i)
        if ds_cnt == 0:
            ax.set_title("Input data")
        # Plot the data points
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright,
                edgecolors='k')

        # Set x- and y-limits on plot and x- and y-ticks for visualization aid 
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        
        i += 1
        # Iterate over classifiers
        for name, clf in zip(names, cluster_algs):
            #determine subplot of interest and initiate plot
            ax = plt.subplot(len(datasets), len(cluster_algs) + 1, i)

            # start clock for timing algorithms
            t0 = time.time()

            # fit clustering algorithm to the data
            clf.fit(X)
            # cluster the data and get cluster labels, and time it
            t1 = time.time()
            if hasattr(clf, 'labels_'):
                y_pred = clf.labels_.astype(np.int)
            else:   
                y_pred = clf.predict(X)

            # Plot the data points colored by clusters
            ax.scatter(X[:, 0], X[:, 1], c=y_pred, cmap=cm_bright,
                    edgecolors='k')

            # Set x- and y-limits on plot and x- and y-ticks for visualization aid
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            # Add title names and put accuracy score in corner of the subplot 
            if ds_cnt == 0:
                ax.set_title(name)
            # plot the computation time on the figure
            ax.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                    transform=plt.gca().transAxes, size=15, horizontalalignment='right')
            # Go to next classifier/data-set combination 
            i += 1
    
    # Finally, show the results. 
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()



