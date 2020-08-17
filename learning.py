# Train the machine learning models and evaluate their performance

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore') # turn off warnings
from sklearn.model_selection import GridSearchCV, StratifiedKFold
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
import helper 



#%%

all_X_trn = []
all_y_trn = []
all_X_tst = []
all_y_tst = []

## Load the data

X_train_0 = pd.read_csv("X_train_0")
all_X_trn.append(X_train_0)


X_train_1 = pd.read_csv("X_train_1")
all_X_trn.append(X_train_1)


X_test_0 = pd.read_csv("X_test_0")
all_X_tst.append(X_test_0)


X_test_1 = pd.read_csv("X_test_1")
all_X_tst.append(X_test_1)


y_train_0 = pd.read_csv("y_train_0")
y_train_0 = (y_train_0.to_numpy()).reshape(len((y_train_0),))
all_y_trn.append(y_train_0)

y_train_1 = pd.read_csv("y_train_1")
y_train_1 = (y_train_1.to_numpy()).reshape(len((y_train_1),))
all_y_trn.append(y_train_1)

y_test_0 = pd.read_csv("y_test_0")
y_test_0 = (y_test_0.to_numpy()).reshape(len((y_test_0),))
all_y_tst.append(y_test_0)

y_test_1 = pd.read_csv("y_test_1")
y_test_1 = (y_test_1.to_numpy()).reshape(len((y_test_1),))
all_y_tst.append(y_test_1)


#%%
# based on accuracy, dropping this column makes the models performance worse on average
# dropped MFCC_10

#all_X_trn = []
#all_y_trn = []
#all_X_tst = []
#all_y_tst = []
#
#
## Rename column titles and add everything to a list called data
#X_train_0 = pd.read_csv("X_train_0_drop")
##X_train_0 = X_train_0.to_numpy()
#all_X_trn.append(X_train_0)
#
#
#X_train_1 = pd.read_csv("X_train_1_drop")
##X_train_1 = X_train_1.to_numpy()
#all_X_trn.append(X_train_1)
#
#
#X_test_0 = pd.read_csv("X_test_0_drop")
##X_test_0 = X_test_0.to_numpy()
#all_X_tst.append(X_test_0)
#
#
#X_test_1 = pd.read_csv("X_test_1_drop")
##X_test_1 = X_test_1.to_numpy()
#all_X_tst.append(X_test_1)
#
#
#y_train_0 = pd.read_csv("y_train_0")
#y_train_0 = (y_train_0.to_numpy()).reshape(len((y_train_0),))
#all_y_trn.append(y_train_0)
#
#y_train_1 = pd.read_csv("y_train_1")
#y_train_1 = (y_train_1.to_numpy()).reshape(len((y_train_1),))
#all_y_trn.append(y_train_1)
#
#y_test_0 = pd.read_csv("y_test_0")
#y_test_0 = (y_test_0.to_numpy()).reshape(len((y_test_0),))
#all_y_tst.append(y_test_0)
#
#y_test_1 = pd.read_csv("y_test_1")
#y_test_1 = (y_test_1.to_numpy()).reshape(len((y_test_1),))
#all_y_tst.append(y_test_1)

#%%
random = 42

# Define models

models = [
            
            # Linear Models
            RidgeClassifier(random_state=random),
            LogisticRegression(multi_class="ovr", random_state=random),
            
            LinearDiscriminantAnalysis(),
            GaussianNB(),
            SVC(random_state=random), 
            MLPClassifier(random_state=random),
            DecisionTreeClassifier(random_state=random),
            KNeighborsClassifier(),
            
            # Ensemble Methods
            RandomForestClassifier(random_state=random),
            GradientBoostingClassifier(random_state=random), # one vs rest
            AdaBoostClassifier(random_state=random),
            
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


bestEstimators = []


#%%

# Model Training
idx = 0
for X_trn_data, y_trn_data, X_tst_data, y_tst_data in zip(all_X_trn, all_y_trn, all_X_tst, all_y_tst):
    print('')
    print('Feature version ' + str(idx))
    idx = idx + 1
    for model, model_name, parameter in zip(models, model_names, parameters):
    
        # Create the pipeline, perform feature scaling first
        pipeline = Pipeline([
                ('scaler', StandardScaler()), 
                (model_name, model)])
         
        # Create the grid search
        # Stratified to keep % of samples in each class in each fold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random) #5 fold cross validation   
        grid = GridSearchCV(pipeline, param_grid=parameter, cv=cv)
        grid.fit(X_trn_data, y_trn_data)
    
        # Get the test accuracy
        score = grid.score(X_tst_data, y_tst_data)
        y_tst_predict = grid.predict(X_tst_data)

        
        # Model Evaluation
        # Print the results
        print('')
        # Test accuracy 
        print(model_name + ' accuracy = %3.5f' %(score))
        print(grid.best_params_)
        
        # Precision, Recall, Fscore
        print('precision, recall, fscore = ')
        print(precision_recall_fscore_support(y_tst_data, y_tst_predict, average='macro'))
        
        # Learning Curves
        helper.plotLearningCurve(grid.best_estimator_, X_trn_data, y_trn_data, cv, model_name) 
        
        # Confusion Matrices
        helper.plotConfMatrix(y_tst_data, y_tst_predict, model_name)
        
        bestEstimators.append(grid.best_estimator_) # Save the best estimator from each model
        
#%%     

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


