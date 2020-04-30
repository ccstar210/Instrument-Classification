import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

n_mfcc = 12
directory = "samples/"

def extractFeatures(filename):
    # load the file
    x, sr = librosa.load(directory + filename)
    
    # Extract the MFCCs
    mfcc = librosa.feature.mfcc(x, sr=sr, n_mfcc=n_mfcc).T        
    return mfcc


def formatData(X, y):
    # Format the MFCC data
    X_data = []
    y_data = []
    for sample, label in zip(X, y):
        features = extractFeatures(sample)
        X_data.append(features.mean(0).reshape((1, n_mfcc)))
        y_data.append(label)
    X_data = np.reshape(X_data, (-1, n_mfcc))
        
    return np.array(X_data), np.array(y_data)


def plotConfMatrix(y_test, y_predict, modelType):
    #Confusion matrix
    conf = np.zeros((11,11), dtype=int)
    titles = ["Cello", "Church Organ", "Clarinet", "Flute", "Guitar", "Harp", "Marimba", "Perldrop", "Piano", "Synlead3", "Violin"]
    for hit in range(len(y_test)):
       conf[y_test[hit]][y_predict[hit]] += 1
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8,8))
    sn.set(font_scale=1.5)
    sn.heatmap(conf, annot=True, fmt='d', ax=ax, cmap="YlGnBu", xticklabels=titles, yticklabels=titles)
    ax.set_ylim(len(conf),0)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix with sklearn for ' + modelType)
    plt.show()


# Load the data
dataFile = "data.csv"
data = pd.read_csv(dataFile, header=0)
X = data.iloc[:,0]
y = data.iloc[:,1]

# Separate the training data into training and false test set
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_trn_data, y_trn_data = formatData(X_trn, y_trn)
X_tst_data, y_tst_data = formatData(X_tst, y_tst)

#%%
# Use cross validation grid search
# TODO: add random forest and gradient boosting

# Define models
models = [SVC(),
          MLPClassifier(),
          DecisionTreeClassifier(),
          MultinomialNB(),
          KNeighborsClassifier()]

model_names = ['SVM',
               'MLP',
               'DT',
               'NB',
               'kNN']

# Define hyperparameters to search
C_range = np.arange(-1.0, 1.0) # np.arange(-4.0, 7.0)
C_values = np.power(10.0, C_range)
gamma_range = np.arange(-1.0, 1.0) # np.arange(-3.0, 2.0)
gamma_values = np.power(10.0, gamma_range)
param_SVM = {'SVM__C':C_values, 'SVM__gamma':gamma_values}

alpha_range = np.arange(-1.0, 0.0) # np.arange(2.0, 2.0, 1.0)
alpha_values = np.power(10.0, C_range)
param_MLP = {'MLP__alpha':alpha_values}

param_DT = {'DT__max_depth':np.arange(1, 2)} # np.arange(1, 11)
param_NB = {'NB__alpha':np.arange(1,2)/100} # np.arange(1,10)/100
param_kNN = {'kNN__n_neighbors':np.arange(10,11)} # np.arange(1,30,2)

parameters = [param_SVM,
              param_MLP,
              param_DT,
              param_NB,
              param_kNN]

scalers = [StandardScaler(),
           StandardScaler(),
           StandardScaler(),
           MinMaxScaler(),
           StandardScaler()]
 
for model, model_name, parameter, scaler in zip(models, model_names, parameters, scalers):
    # Create the pipeline
    pipeline = Pipeline([('scaler', scaler), (model_name, model)])
     
    # Create the grid search
    grid = GridSearchCV(pipeline, param_grid=parameter, cv=5)    
    grid.fit(X_trn_data, y_trn_data)

    # Get the accuracy         
    print(model_name + " score = %3.2f" %(grid.score(X_tst_data, y_tst_data)))
    print(grid.best_params_)
    #plotConfMatrix(y_tst_data, y_predict, model_name)
    
