# Instrument/Timbre Classification using Machine Learning

## Introduction
Audio-based instrument classifier that identifies an instrument based on its timbre given a short sample of music.

## Technologies
* Python: 3.7.4
* Python libraries: numpy, matplotlib, pandas, seaborn, sklearn, librosa, soundfile

## Table of Contents
* [Background](#background)
* [Dataset](#dataset)
* [Data Preprocessing](#data-preprocessing)
* [Machine Learning Methods](#machine-learning-methods)
* [Evaluation](#evaluation)
* [Sources](#sources)

## Background
Timbre distinguishes the type of instrument even when the same pitch is played due to each instrument's unique frequency characteristics. The instruments tested were cello, church organ, clarinet, flute, guitar, harp, marimba, pinao, violin, and synthetic effects (PerlDrop and SynLead3).

## Dataset
An electronic keyboard that emulated various instruments was used to record music samples from various genres.\
See the Recordings folder for the recordings in .wav files.


## Data Preprocessing
### Data cleaning
The recordings were converted from stereo to mono and resampled to 16kHz to reduce file size. Silent beginning and ending periods of each sample were removed. Each sample was split into 2 second blocks with a 1 second step size between the blocks.\
See the samples folder created from the create_samples.py file\
To make reading the data in easier, data.csv was created from create_data_file.py

### Feature Extraction
Mel-frequency cepstral coefficients (MFCCs) were used as the features. They are commonly used in speech recognition and for processing music.\
See feature_extraction.py

## Machine Learning Methods
The following 11 machine learning algorithms were selected: 
 * Ridge classification
 * Logistic regression
 * Linear discriminant analysis
 * Gaussian naive bayes
 * Support vector machine
 * Multi-layer perceptron
 * Decision trees 
 * k-nearest neighbors
 * Random forest
 * Gradient boosting
 * AdaBoost
 See learning.py for training the models

## Evaluation 
Evalution metrics used were:
 * Test accuracy
 * Precision
 * Recall
 * F-score
 * Learning curves
 * Confusion matrices
 * ROC curves/ AUC 
 See learning.py for test accuracies, precision, recall, and F-score.\
 See images folder for learning curves, confusion matrices, and ROC curves/AUC\
 The top performing models were support vector machine 

## Sources
Feature extraction based on:
https://musicinformationretrieval.com/genre_recognition.html?fbclid=IwAR0QnFEJi2pXzll7unDKQR9GS5RtnJelA42d9ijcax-2Wx6n_LYpLj58r1M
