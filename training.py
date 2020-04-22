#creates the training data with labels from the sample folder, trains the classifier, and predicts the label for test data

import librosa
import matplotlib.pyplot as plt
import IPython.display as ipd
import librosa.display
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

import seaborn as sn

def detLabel(filename):
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
    return label
    

def prepareSample(filename):
    
    #load the file
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
    #mfcc_scaled.mean(axis=0)
    #mfcc_scaled.std(axis=0)
    
    return mfcc_scaled

def plotConfMatrix(y_test,y_predict, modelType):
    #Confusion matrix
    conf = np.zeros((11,11), dtype=int)
    titles = ["Cello","Church Organ", "Clarinet", "Flute", "Guitar", "Harp", "Marimba", "Perldrop", "Piano", "Synlead3", "Violin"]
    for hit in range(len(x_test)):
       conf[y_test[hit]][y_predict[hit]] += 1
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8,8))
    sn.set(font_scale=1.5)
    sn.heatmap(conf, annot=True, fmt='d', ax=ax, cmap="YlGnBu", xticklabels=titles, yticklabels=titles)
    ax.set_ylim(len(conf),0)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix with sklearn for '+ modelType)
    plt.show()

fileOrigin = "samples1"
fileTest = "test"

i=1

#iterate through the samples folder to get each 2 sec recording
for recording in os.listdir(fileOrigin): 

    filename = "samples1/" + recording

    mfcc_scaled = prepareSample(filename)
    
    #add recording data to the x_train array (features)
    if i==1:
        x_train = mfcc_scaled
    else:
        x_train = np.vstack([x_train,mfcc_scaled])

    
    #determine the label of the recording
    label = detLabel(filename)

        
    #add recording label to the y_train array
    if i==1:
        y_train = np.full((len(mfcc_scaled), 1), label)
    else:
        y_train = np.vstack((y_train,np.full((len(mfcc_scaled),1 ), label)))
    

    i = 2
y_train = y_train.reshape((len(y_train),))

#print(x_train) #gives (27144,12), each recording is an array of 87 rows with 312 recordings, 87*312=27144, 12 for the # of features
#print(y_train.shape) #gives (27144,)

#create test data
#iterate through the test folder to get each 2 sec recording
j=1
for recording in os.listdir(fileTest): 

    filename = "test/" + recording

    mfcc_scaled = prepareSample(filename)
    
    #add recording data to the x_train array (features)
    if j==1:
        x_test = mfcc_scaled
    else:
        x_test = np.vstack([x_test,mfcc_scaled])
        
        
    #determine the label of the recording
    label = detLabel(filename)

        
    #add recording label to the y_train array
    if j==1:
        y_test = np.full((len(mfcc_scaled), 1), label)
    else:
        y_test = np.vstack((y_test,np.full((len(mfcc_scaled),1 ), label)))
    j=2
    
#convert y_test to right shape    
y_test = y_test.reshape((len(y_test),))        

#train the classifier

#create classifer model object
modelSVM = SVC()
#modelNN = MLPClassifier( alpha=1e-5,hidden_layer_sizes=(5,), random_state=1)
#modelDT = DecisionTreeClassifier(criterion='entropy', max_depth=20)
#modelRF = RandomForestClassifier(criterion='entropy', max_depth=30) #higher max_depth gives better accuracy
#modelNB = MultinomialNB()
#modelkNN = KNeighborsClassifier(n_neighbors=100, weights='distance')
#modelGB = GradientBoostingClassifier()


#train classifer
modelSVM.fit(x_train, y_train)
#modelNN.fit(x_train, y_train)
#modelDT.fit(x_train, y_train)
#modelRF.fit(x_train, y_train)
#modelNB.fit(x_train, y_train)
#modelkNN.fit(x_train, y_train)
#modelGB.fit(x_train, y_train)

#predict
y_predict_SVM = modelSVM.predict(x_test)
#y_predict_NN = modelNN.predict(x_test)
#y_predict_DT = modelDT.predict(x_test)
#y_predict_RF = modelRF.predict(x_test)
#y_predict_NB = modelNB.predict(x_test)
#y_predict_kNN = modelkNN.predict(x_test)
#y_predict_GB = modelGB.predict(x_test)


print(modelSVM.score(x_test,y_test))
#print(modelNN.score(x_test,y_test))
#print(modelDT.score(x_test,y_test))
#print(modelRF.score(x_test,y_test))
#print(modelNB.score(x_test,y_test))
#print(modelkNN.score(x_test, y_test))
#print(modelGB.score(x_test, y_test))

#Confusion matrix
plotConfMatrix(y_test,y_predict_SVM, "SVM")
#plotConfMatrix(y_test,y_predict_NN, "Neural Networks")
#plotConfMatrix(y_test,y_predict_DT, "Decision Trees")
#plotConfMatrix(y_test,y_predict_RF, "Random Forest")
#plotConfMatrix(y_test,y_predict_NB, "Naive Bayes")
#plotConfMatrix(y_test, y_predict_kNN, "Nearest Neighbors")
#plotConfMatrix(y_test, y_predict_GB, "Gradient Boosting")



