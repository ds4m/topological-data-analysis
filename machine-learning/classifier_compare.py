#!/bin/env python 

"""
Classification comparison in Python

Code taken and modified from: http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

Additional resources:
    -sklearn user tutorial: http://scikit-learn.org/stable/user_guide.html
    -machine learning with scikit-learn tutorial: http://scikit-learn.org/stable/tutorial/basic/tutorial.html
    -choosing the right estimator: http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

Melissa McGuirl, Brown University. 
Contact: melissa_mcguirl@brown.edu

Date Modified: 07/23/18

As always, we first load in the modules we need to use. 
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def main():
    

    # Create vector of names for  algorithms of interest 
    names = ["K=Nearest Neighbors K=15","K-Nearest Neighbors K=10","K-Nearest Neighbors K=15",
                "Linear SVM", "RBF SVM",  "Poly-4 SVM"]    
    
    
    # Make a vector of classifiers that we want to test. Specify input
    # parameters for each classifier. Here we are testing KNN with K=5,10,15 as
    # specified by the function input, and a support vector machine classifier
    # using a linear kernel, rbf kernel, and degree 4 polynomial kernel. For the
    # SVM classifiers we specify C=2, meaning that we want to partition our data
    # into two classes. 
    classifiers = [KNeighborsClassifier(5),KNeighborsClassifier(10),KNeighborsClassifier(15),
            SVC(kernel="linear", C=2),SVC(kernel='rbf', C=2), SVC(kernel="poly", degree=4, C=2)]


    # First, make a toy linearly separable data set 
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                    random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X,y)

    # Consider three data sets from Python: data sampled from two noisy moon
    # shapes, data sampled from an inner and outer noisy circle, and noisy linearly
    # separable data generated above. 
    datasets = [make_moons(noise=0.3, random_state=0),
                make_circles(noise=0.2, factor=0.5, random_state=1),
                linearly_separable ]

    # Initiate figure for plotting results 
    figure = plt.figure(figsize=(27, 9))
    
    h = 0.2  #Define spatial step size
    i = 1   #Counting variable for plotting 

    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):
        # Preprocess dataset, split into training and test part
        X, y = ds
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)
        
        # Create a mesh over domain     
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))

        # Plot the three datasets first
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        if ds_cnt == 0:
            ax.set_title("Input data")
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                edgecolors='k')
        # Plot the  testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
                edgecolors='k')

        # Set x- and y-limits on plot and x- and y-ticks for visualization aid 
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        
        i += 1
        # Iterate over classifiers
        for name, clf in zip(names, classifiers):
            #determine subplot of interest and initiate plot
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

            # Fit training data using current algorithm
            clf.fit(X_train, y_train)

            # Now use the test data to determing a score for how accurate the
            # algorithm classified the testing set. Here we use the model
            # trained on the testing set and apply it to the test set. The score 
            # is the mean accuracy of the algorithn applied to the test data. 
            # For more information on how to evaluate a machine learning
            # algorithm applied to a given data set see: http://scikit-learn.org/stable/modules/model_evaluation.html
            score = clf.score(X_test, y_test)

            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            if hasattr(clf, "decision_function"):
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:   
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

            # Plot the training points
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                    edgecolors='k')
            # Plot testing points
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                    edgecolors='k', alpha=0.6)
            # Set x- and y-limits on plot and x- and y-ticks for visualization aid
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            # Add title names and put accuracy score in corner of the subplot 
            if ds_cnt == 0:
                ax.set_title(name)
            ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                    size=15, horizontalalignment='right')
            # Go to next classifier/data-set combination 
            i += 1
    
    # Finally, show the results. 
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()



