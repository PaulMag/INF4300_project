# INF4300 Digital image analysis

This repository contains code used for the two mandatory exercies in 
"INF4300 - Digital image analysis" at the University of Oslo.

## glcm.py
Program for making GLCMs of images, and various feature maps by making GLCMs on gliding
windows over an input image.
This program also does small things like showing the input image and its histogram,
and the weight maps for the various GLCM features.
All products can be plotted and displayed and/or saved to a file,
both as PNG-images and Python arrays (in pickle-files).
glcm.py is controlled by cmd-line args.

## segment.py
Small program that segment images into regions by simple treshold limits.
Its intended use is to segment the feature maps produced by glcm.py.
segment.py is controlled by cmd-line args.

## classifier.ipynb
"Interactive" program in form if an iPython Notebook.
Implements a multivariate Gaussian classifier.
Loads several feature maps from a set of training data and then the corresponding
features from the data you want to classify.
Training is performed on the training data and then likelihoods are computed on 
the test data which is used to classify all pixels.
Use the program by setting the filenames of all the feature images you want
to load in the "Loading"-section and then run all the code blocks.
