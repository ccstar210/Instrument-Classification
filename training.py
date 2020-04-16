#creates the training data with labels from the sample folder

import librosa
import matplotlib.pyplot as plt
import IPython.display as ipd
import librosa.display
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
from sklearn.svm import SVC

fileOrigin = "samples"

i=1

#iterate through the samples folder to get each 2 sec recording
for recording in os.listdir(fileOrigin): 
    #filename1 = recording
    #load the file
    filename = "samples/" + recording
    x, sr = librosa.load(filename)
    
    #ipd.Audio(x, rate=sr)
    
    #time-domain waveform
    #plt.figure()
    #librosa.display.waveplot(x, sr)
    
    #power melspectrogram
    #S = librosa.feature.melspectrogram(x, sr=sr, power=2.0)
    #convert the amplitude to decibels
    #Sdb = librosa.power_to_db(S)
    
    #Get the 12 features (MFCCs)
    n_mfcc = 12
    mfcc = librosa.feature.mfcc(x, sr=sr, n_mfcc=n_mfcc).T
    #tranpose where a column is the feature and a row is a training example
    #mfcc.shape
    #mean
    #mfcc.mean(axis=0)
    #standard devation?
    #mfcc.std(axis=0)
    
    #make features have zero mean and unit variance
    scaler = StandardScaler()
    mfcc_scaled = scaler.fit_transform(mfcc)
    mfcc_scaled.mean(axis=0)
    mfcc_scaled.std(axis=0)
    
    #add recording data to the x_train array (features)
    if i==1:
        x_train = mfcc_scaled
    else:
        x_train = np.vstack([x_train,mfcc_scaled])

    
    #determine the label of the recording
    if "cello" in filename: 
        label = 0
    elif "church" in filename:
        label = 1
    elif "clarinet" in filename:
        label = 2
    elif "flute" in filename:
        label = 3
    elif "guitar" in filename:
        label = 4
    elif "harp" in filename:
        label = 5
    elif "marimba" in filename:
        label = 6
    elif "perldrop" in filename:
        label = 7
    elif "piano" in filename:
        label = 8
    elif "synlead3" in filename:
        label = 9
    else: #violin
        label = 10
        
    #add recording label to the y_train array
    if i==1:
        y_train = np.full((len(mfcc_scaled), 1), label)
    else:
        y_train = np.vstack((y_train,np.full((len(mfcc_scaled),1 ), label)))
    

    i = 2
y_train = y_train.reshape((len(y_train),))

#print(x_train) #gives (27144,12), each recording is an array of 87 rows with 312 recordings, 87*312=27144, 12 for the # of features
#print(y_train.shape) #gives (27144,)


#train the classifier

#create classifer model object
modelSVM = SVC()

#train classifer
modelSVM.fit(x_train, y_train)
