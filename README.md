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
An electronic keyboard that emulated various instruments was used to record music samples from various genres.

## Data Preprocessing
### Data cleaning
The recordings were converted from stereo to mono and resampled to 16kHz to reduce file size. Silent beginning and ending periods of each sample were removed. Each sample was split into 2 second blocks with a 1 second step size between the blocks.

### Feature Extraction
Mel-frequency cepstral coefficients (MFCCs) were used as the features.

## Machine Learning Methods
The following 6 machine learning algorithms were selected: 
 * Support vector machines
 * Multi-layer perceptron
 * Decision trees 
 * k-nearest neighbors
 * Random forest
 * Gradient boosting

## Evaluation 
Evalution metrics used were cross-validation accuracy, test accuracy, precision, recall, F-score, ROC curves, AUC, learning curves, confusion matrices.

## Sources
Feature extraction based on:
https://musicinformationretrieval.com/genre_recognition.html?fbclid=IwAR0QnFEJi2pXzll7unDKQR9GS5RtnJelA42d9ijcax-2Wx6n_LYpLj58r1M
