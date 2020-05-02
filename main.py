import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import warnings
from itertools import cycle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


n_mfcc = 12
directory = "samples/"
warnings.filterwarnings('ignore') # turn off warnings
random = 42

def extractFeatures(filename):
    # load the file
    x, sr = librosa.load(directory + filename)
    
    # Extract the MFCCs
    mfcc = librosa.feature.mfcc(x, sr=sr, n_mfcc=n_mfcc).T        
    return mfcc


def formatData(X, y, i):
    # Format the MFCC data
    X_data = []
    y_data = []
    if i==0:
        for sample, label in zip(X, y):
            features = extractFeatures(sample)
            mfcc = np.mean(features, axis=0).reshape((1, n_mfcc))
            X_data.append(mfcc)
            y_data.append(label)
        X_data = np.reshape(X_data, (-1, n_mfcc))
    else:
        for sample, label in zip(X, y):
            features = extractFeatures(sample)
            mfcc = np.mean(features, axis=0).reshape((1, n_mfcc))
            var = np.var(features, axis=0).reshape((1, n_mfcc))
            X_data.append(np.hstack((mfcc, var)))
            y_data.append(label)
        X_data = np.reshape(X_data, (-1, 2*n_mfcc))
        
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
    
def plotLearningCurve(estimator, X, y, cv, modelType):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, shuffle=True, random_state=random)
    train_scores = 100*train_scores
    test_scores = 100*test_scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    fig, ax = plt.subplots(figsize=(8,8))
    plt.grid()
    plt.plot(train_sizes, train_scores_mean, label = 'Training accuracy')
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1)    
    plt.plot(train_sizes, test_scores_mean, label = 'Cross-validation accuracy')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1)
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Training set size')
    plt.title('Learning Curves with sklearn for ' + modelType)
    plt.legend(loc="best")
    plt.show()

#based on code from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
def plotROCCurve(estimator, X_train, y_train, X_test, y_test, modelType):
    #binarize output
    y_train_bin = label_binarize(y_train, classes=[0,1,2,3,4,5,6,7,8,9,10])
    y_test_bin = label_binarize(y_test, classes=[0,1,2,3,4,5,6,7,8,9,10])
    numClasses = y_train_bin.shape[1]
    #one vs all classification
    classifier = OneVsRestClassifier(estimator)
    y_score = classifier.fit(X_train, y_train_bin).decision_function(X_test)
    #print(y_test_bin.shape)
    #print(y_score.shape)
    
    #get the ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(numClasses):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:,i], y_score[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
      
    print('hello')
    #get micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    print('hello1')
    #plot ROC curve for all classes
    fig, ax = plt.subplots(figsize=(8,8))
    plt.grid()
    lw=2
    plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)
    print('hello2')
    colors = cycle(['red','goldenrod','darkorange', 'yellow', 'olive', 'lime', 'green','cornflowerblue', 'aqua','blue', 'darkviolet'])
    for i, color in zip(range(numClasses), colors):
        
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
        

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlabel('False Positives Rate')
    plt.ylabel('True Positives Rate')
    plt.title('ROC Curve with sklearn for ' + modelType)
    plt.legend(loc="best")
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
all_X_trn = []
all_y_trn = []
all_X_tst = []
all_y_tst = []
numFeatureSettings = 2
print('Extracting features...')
for i in range(numFeatureSettings):
    X_trn_data, y_trn_data = formatData(X_trn, y_trn, i)
    X_tst_data, y_tst_data = formatData(X_tst, y_tst, i)
    all_X_trn.append(X_trn_data)
    all_y_trn.append(y_trn_data)
    all_X_tst.append(X_tst_data)
    all_y_tst.append(y_tst_data)
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
param_MLP = {'MLP__alpha' : np.power(10.0, np.arange(-3.0, 1.0))} #'MLP_layers': np.linspace((5,),(100,),5)
param_DT = {'DT__max_depth' : np.arange(1, 20, 2)}
param_NB = {'NB__alpha' : np.arange(1, 10)/100}
param_kNN = {'kNN__n_neighbors' : np.arange(1, 10, 2)}
param_RF = {'RF__n_estimators' : np.arange(100, 500, 200),
            'RF__max_depth' : np.arange(1, 20, 2)}
param_GB = {'GB__learning_rate': np.arange(0.1, 0.5, 0.1)} #'GB__n_estimators' : np.arange(50, 200, 50),

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

idx = 0
for X_trn_data, y_trn_data, X_tst_data, y_tst_data in zip(all_X_trn, all_y_trn, all_X_tst, all_y_tst):
    print('')
    print('Feature version ' + str(idx))
    idx = idx + 1
    for model, model_name, parameter, scaler in zip(models, model_names, parameters, scalers):
        # Create the pipeline
        pipeline = Pipeline([('scaler', scaler), (model_name, model)])
         
        # Create the grid search
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random) #5 fold cross validation   
        grid = GridSearchCV(pipeline, param_grid=parameter, cv=cv)
        grid.fit(X_trn_data, y_trn_data)
    
        # Get the accuracy
        score = grid.score(X_tst_data, y_tst_data)
        y_tst_predict = grid.predict(X_tst_data)
        
        # Print the results
        #print('')
        print(model_name + ' accuracy = %3.2f' %(score))
        #print(grid.best_params_)
        #print('precision, recall, fscore = ')
        #print(precision_recall_fscore_support(y_tst_data, y_tst_predict, average='macro'))
        
        #plotLearningCurve(grid.best_estimator_, X_trn_data, y_trn_data, cv, model_name)        
        #plotConfMatrix(y_tst_data, y_tst_predict, model_name)
        plotROCCurve(grid.best_estimator_, X_trn_data, y_trn_data, X_tst_data, y_tst_data, model_name)
    
