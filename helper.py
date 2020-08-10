import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import warnings
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc




n_mfcc = 12
directory = "samples/"
warnings.filterwarnings('ignore') # turn off warnings
random = 42



# Helper functions

# Extract the MFCC features
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
        #  12 features (MFCC)
        for sample, label in zip(X, y):
            features = extractFeatures(sample)
            mfcc = np.mean(features, axis=0).reshape((1, n_mfcc)) # mean
            X_data.append(mfcc)
            y_data.append(label)
        X_data = np.reshape(X_data, (-1, n_mfcc))
    else:
        # 24 features (MFCC + variance)
        for sample, label in zip(X, y):
            features = extractFeatures(sample)
            mfcc = np.mean(features, axis=0).reshape((1, n_mfcc)) # mean
            var = np.var(features, axis=0).reshape((1, n_mfcc))  # added variance
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


def plotROCCurve(estimator, X_train, y_train, X_test, y_test, modelType):
    y_train_bin = label_binarize(y_train, classes=[0,1,2,3,4,5,6,7,8,9,10])
    y_test_bin = label_binarize(y_test, classes=[0,1,2,3,4,5,6,7,8,9,10])
    
    classifier = OneVsRestClassifier(estimator)
    if hasattr(classifier, "decision_function"):
        y_score = classifier.fit(X_train, y_train_bin).decision_function(X_test)
    else:
        y_score = classifier.fit(X_train, y_train_bin).predict_proba(X_test)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # get micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    plt.plot(fpr["micro"], tpr["micro"], label=modelType + ' (area = {0:0.4f})'
               ''.format(roc_auc["micro"]), linewidth=4, alpha=0.7)