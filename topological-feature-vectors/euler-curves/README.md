# EulerCurves
Computing Euler Characteristic Curves for Greyscale Images (MNIST and Devanagari Characters)

This project contains python code to use the SECT (Smooth Euler Characteristic Transform) to compare images.
For some theoretical results on the SECT, see: https://arxiv.org/pdf/1611.06818.pdf.

Data
----
1. MNIST data set (in CSV format) can be found here: https://pjreddie.com/projects/mnist-in-csv/
2. An MNIST-like data set of Devanagari characters can be found here: https://archive.ics.uci.edu/ml/datasets/Devanagari+Handwritten+Character+Dataset 


Files
-----
1. mnist.py -- this loads the MNIST CSV data set into python, organizing it into arrays of greyscale images and labels. Then the prediction accuracy is measured for a choice of directions. This file can be edited to change the size of the loaded training and testing data, as well as the directions.
2. devanagari.py -- this loads the Devanagari directory into python, organizing it into arrays of greyscale images and labels.  Then the prediction accuracy is measured for a choice of directions. This file can be edited to change the size of the loaded training and testing data, as well as the directions.
3. ECC.py -- this contains the functions that build the cubical complexes for greyscale images, produce euler characteristic curves, smoothes and compares them, predicts labels (using KNN), etc.

Notes
-----
1. There is room for streamlining the data, tuning the computation pipeline, and building a more complex prediction model.
2. Performance on Default Settings:

MNIST -- 200 training images/100 testing images, 12 directions, runs 1-2 mins, 86% accuracy

Devanagari -- (Using first 10 letters) 20 training images per letter/10 testing images per letter, 3 directions, runs 1-2 minutes, 76% accuracy. Using more letters requires more training time.

Accuracy can improved by adding more directions, a larger training set, and adjusting the classifier.

For any questions, contact Elchanan Solomon at ysolomon AT math DOT brown DOT edu.
