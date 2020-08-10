import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

import helper # put all the helper functions in the file called helper.py

random = 42

#%%
# Load the data
dataFile = "data.csv"
data = pd.read_csv(dataFile, header=0)
X = data.iloc[:,0]
y = data.iloc[:,1]

# Separate the training data into training and false test set
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

#%%
# Extract features (MFCC's)

all_X_trn = []
all_y_trn = []
all_X_tst = []
all_y_tst = []

# Investigating 2 types of feature sets
numFeatureSettings = 2

print('Extracting features...')
for i in range(numFeatureSettings):
    X_trn_data, y_trn_data = helper.formatData(X_trn, y_trn, i)
    X_tst_data, y_tst_data = helper.formatData(X_tst, y_tst, i)
    all_X_trn.append(X_trn_data)
    all_y_trn.append(y_trn_data)
    all_X_tst.append(X_tst_data)
    all_y_tst.append(y_tst_data)
print('Feature extraction complete')


#%%

# Define models
models = [
            
            # Linear Models
            RidgeClassifier(),
            LogisticRegression(multi_class="ovr"),
            
            LinearDiscriminantAnalysis(),
            GaussianNB(),
            SVC(), # one vs one
            MLPClassifier(),
            DecisionTreeClassifier(),
            KNeighborsClassifier(),
            # Ensemble Methods
            RandomForestClassifier(),
            GradientBoostingClassifier(), # one vs rest
            AdaBoostClassifier(),
            
          ]

model_names = [
                'Ridge',
                'LogReg',
                'LDA',
                'NB',
                'SVM',
                'MLP',
                'DT',
                'kNN',
                'RF',
                'GB',
                'AB',
                
               ]

# Define hyperparameters to search
param_Ridge = {'Ridge__alpha': [0.1,1.0,10.0]},
param_LogReg = {'LogReg__solver': ['newton-cg','sag','saga', 'lbfgs']}, #'LogReg__penalty': ['l2','elasticnet','none'],
param_LDA = {'LDA__solver': ['svd', 'lsqr', 'eigen']}
param_NB = {},
param_SVM = {'SVM__C': np.power(10.0, np.arange(-1.0, 4.0)),
             'SVM__gamma': np.power(10.0, np.arange(-3.0, 1.0))}
param_MLP = {'MLP__alpha': np.power(10.0, np.arange(-3.0, 1.0))} #'MLP_layers': np.linspace((5,),(100,),5)
param_DT = {'DT__max_depth': np.arange(1, 20, 2)}
param_kNN = {'kNN__n_neighbors': np.arange(1, 10, 2)}
param_RF = {'RF__n_estimators': np.arange(100, 500, 200),
            'RF__max_depth': np.arange(1, 20, 2)}
param_GB = {'GB__learning_rate': np.arange(0.1, 0.5, 0.1),
            'GB__max_depth': np.arange(1,6)}, #'GB__n_estimators' : np.arange(50, 200, 50),
param_AB = {'AB__n_estimators': np.arange(50, 200, 50),
            'AB__learning_rate': np.arange(0.1,0.5,0.1)},



parameters = [
                param_Ridge,
                param_LogReg,
                param_LDA,
                param_NB,
                param_SVM,
                param_MLP,
                param_DT,
                param_kNN,
                param_RF,
                param_GB,
                param_AB,
                
              ]

#scalers = [MinMaxScaler(),
#           MinMaxScaler(),
#           MinMaxScaler(),
#           MinMaxScaler(),
#           MinMaxScaler(),
#           MinMaxScaler()]
scalers = [
            StandardScaler(),
            StandardScaler(),
            StandardScaler(),
            StandardScaler(),
            StandardScaler(),
            StandardScaler(),
           ]

idx = 0
bestEstimators = []




#%%
#for X_trn_data, y_trn_data, X_tst_data, y_tst_data in zip(all_X_trn, all_y_trn, all_X_tst, all_y_tst):
print('')
print('Feature version ' + str(idx))
#idx = idx + 1
for model, model_name, parameter in zip(models, model_names, parameters):

    # Create the pipeline
    pipeline = Pipeline([
            ('scaler', StandardScaler()), 
            (model_name, model)])
     
    # Create the grid search
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random) #5 fold cross validation   
    grid = GridSearchCV(pipeline, param_grid=parameter, cv=cv)
    grid.fit(X_trn_data, y_trn_data)

    # Get the test accuracy
    score = grid.score(X_tst_data, y_tst_data)
    y_tst_predict = grid.predict(X_tst_data)
    
    # Print the results
    print('')
    print(model_name + ' accuracy = %3.5f' %(score))
    print(grid.best_params_)
    print('precision, recall, fscore = ')
    print(precision_recall_fscore_support(y_tst_data, y_tst_predict, average='macro'))
    
    helper.plotLearningCurve(grid.best_estimator_, X_trn_data, y_trn_data, cv, model_name)        
    #helper.plotConfMatrix(y_tst_data, y_tst_predict, model_name)
    bestEstimators.append(grid.best_estimator_) # Save the best estimator from each model
        
#%%     
# Model Evaluation
# Plot each model's best ROC curve for each feature set
idx = 0
fig, ax = plt.subplots(figsize=(18,8))
plt.grid()

for X_trn_data, y_trn_data, X_tst_data, y_tst_data in zip(all_X_trn, all_y_trn, all_X_tst, all_y_tst):
    plt.subplot(1,2,idx+1)
    for estimator, model_name in zip(bestEstimators, model_names):    
        helper.plotROCCurve(estimator, X_trn_data, y_trn_data, X_tst_data, y_tst_data, model_name)
    plt.plot([0, 1], [0, 1], color='navy', lw=4, linestyle='--', alpha=0.7)
    plt.xlabel('False Positives Rate')
    plt.ylabel('True Positives Rate')
    plt.title('ROC curves for feature set ' + str(idx))
    plt.legend(loc="best")    
    plt.tight_layout()
    idx += 1
plt.show()

