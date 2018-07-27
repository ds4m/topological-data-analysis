# -*- coding: utf-8 -*-
"""
Analyzing and Classifying MNIST Dataset using euler characteristic curves

@author: Elchanan Solomon
"""

import ECC
import numpy as np

#Sample 8 directions on the unit circle
#directions = [(np.cos(x),np.sin(x)) for x in np.linspace(0,2*np.pi,8)]
#some sample directions
directions = [(1,0),(0,1),(-1,0),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1),(2,1),(2,-1),(-2,1),(-2,-1)]

#Predict labels for 100 test images
labels = ECC.predict_labels(mnist_train_data,mnist_train_labels,mnist_test_data[:100],directions)
#Display the score
print(sum(labels == mnist_test_labels[:100]))

