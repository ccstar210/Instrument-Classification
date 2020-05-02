import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

n_mfcc = 13
directory = "samples/"
warnings.filterwarnings('ignore') # turn off warnings

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
        mfcc = features.mean(0).reshape((1, n_mfcc))
        var = np.var(features, axis=0).reshape((1, n_mfcc))
        #mfcc_delta = librosa.feature.delta(features).mean(0).reshape((1, n_mfcc))
        #mfcc_delta2 = librosa.feature.delta(features, order=2).mean(0).reshape((1, n_mfcc))
        X_data.append(np.hstack((mfcc, var))) #np.hstack((mfcc, mfcc_delta, mfcc_delta2))
        y_data.append(label)
    X_data = np.reshape(X_data, (-1, 2*n_mfcc)) #3*n_mfcc
        
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


#%%
# Load the data
dataFile = "data.csv"
data = pd.read_csv(dataFile, header=0)
X = data.iloc[:,0]
y = data.iloc[:,1]

# Separate the training data into training and false test set
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

# Extract features
print('Extracting features...')
X_trn_data, y_trn_data = formatData(X_trn, y_trn)
X_tst_data, y_tst_data = formatData(X_tst, y_tst)
print('Feature extraction complete')

#%%
# Use cross validation grid search
# TODO: add random forest and gradient boosting

# Define models
models = [SVC(),
          MLPClassifier(),
          DecisionTreeClassifier(),
          MultinomialNB(),
          KNeighborsClassifier(),
          RandomForestClassifier(),
          GradientBoostingClassifier()]

model_names = ['SVM',
               'MLP',
               'DT',
               'NB',
               'kNN',
               'RF',
               'GB']

# Define hyperparameters to search
param_SVM = {'SVM__C' : np.power(10.0, np.arange(-1.0, 4.0)),
             'SVM__gamma' : np.power(10.0, np.arange(-3.0, 1.0))}
param_MLP = {'MLP__alpha' : np.power(10.0, np.arange(-3.0, 1.0))}
param_DT = {'DT__max_depth' : np.arange(1, 20, 2)}
param_NB = {'NB__alpha' : np.arange(1, 10)/100}
param_kNN = {'kNN__n_neighbors' : np.arange(1, 10, 2)}
param_RF = {'RF__n_estimators' : np.arange(100, 1000, 100)}
param_GB = {'GB__n_estimators' : np.arange(100, 1000, 100)}

parameters = [param_SVM,
              param_MLP,
              param_DT,
              param_NB,
              param_kNN,
              param_RF,
              param_GB]

scalers = [StandardScaler(),
           StandardScaler(),
           StandardScaler(),
           MinMaxScaler(),
           StandardScaler(),
           StandardScaler(),
           StandardScaler()]
 
for model, model_name, parameter, scaler in zip(models, model_names, parameters, scalers):
    # Create the pipeline
    pipeline = Pipeline([('scaler', scaler), (model_name, model)])
     
    # Create the grid search
    grid = GridSearchCV(pipeline, param_grid=parameter, cv=5)    
    grid.fit(X_trn_data, y_trn_data)

    # Get the accuracy
    score = grid.score(X_tst_data, y_tst_data)
    y_tst_predict = grid.predict(X_tst_data)
    
    # Print the results
    print('')
    print(model_name + " accuracy = %3.2f" %(score))
    print(grid.best_params_)
    print('precision, recall, fscore = ')
    print(precision_recall_fscore_support(y_tst_data, y_tst_predict, average='macro'))
    
    #plotConfMatrix(y_tst_data, y_tst_predict, model_name)
    
