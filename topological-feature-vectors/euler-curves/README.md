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
1. load_mnist.py -- this loads the MNIST CSV data set into python, organizing it into arrays of greyscale images and labels. This file can be edited to change the size of the loaded training and testing data.
2. load_devanagari.py -- this loads the Devanagari directory into python, organizing it into arrays of greyscale images and labels. This file can be edited to change the number of images loaded per each of the 36 characters, for both the training and testing data.
3. ECC.py -- this contains the functions that build the cubical complexes for greyscale images, produce euler characteristic curves, smoothes and compares them, predicts labels (using KNN), etc.
4. analayze_MNIST.py -- this is a short file, to be run after load_mnist.py, to compute prediction accuracy along a simple set of directions.
5. analyze_devan.py -- this is a short file, to be run after load_devanagari.py, to compute prediction accuracy along a simple set of directions.

Notes
-----
1. There is room for streamlining the data, tuning the computation pipeline, and building a more complex prediction model.
2. So far, MNIST accuracy tops out at around 93%, and Devangari accuracy tops out at around 80%. Higher percentages can be expected by loading more training data, but the code takes a while to run. 

For any questions, contact Elchanan Solomon at ysolomon AT math DOT brown DOT edu.
