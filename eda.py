# Perform exploratory data analysis on the datasets

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#%% 
    
X_featureset_0 = [] #featureset 0 is the mean MFCCs
X_featureset_1 = [] #featureset 1 is the mean MFCCs and variance
y = []


X_train_0 = pd.read_csv("csv/X_train_0.csv")
X_featureset_0.append(X_train_0)

X_train_1 = pd.read_csv("csv/X_train_1.csv")
X_featureset_1.append(X_train_1)

X_test_0 = pd.read_csv("csv/X_test_0.csv")
X_featureset_0.append(X_test_0)

X_test_1 = pd.read_csv("csv/X_test_1.csv")
X_featureset_1.append(X_test_1)

y_train_0 = pd.read_csv("csv/y_train_0.csv")
y.append(y_train_0)

y_train_1 = pd.read_csv("csv/y_train_1.csv")
y.append(y_train_1)

y_test_0 = pd.read_csv("csv/y_test_0.csv")
y.append(y_test_0)

y_test_1 = pd.read_csv("csv/y_test_1.csv")
y.append(y_test_1)



#%%

# Understand the data and look for missing values
i=0
for df in X_featureset_0 + X_featureset_1 + y: 
    print("*****************************dataset" + str(i) + "***************************************")
    # all numerical values
    print(df.info())
    print(df.describe())
    # no missing values
    print(df.isnull().sum())
    i=i+1


#%%

# Univariate analysis

# Training set for feature set 0 
X_train_0.hist(bins=15, edgecolor='black', linewidth=1.0,
           xlabelsize=8, ylabelsize=8, grid=False)    
plt.tight_layout()   

# Test set for feature set 0
X_test_0.hist(bins=15, edgecolor='black', linewidth=1.0,
           xlabelsize=8, ylabelsize=8, grid=False)    
plt.tight_layout() 

# Training set for feature set 1
X_train_1.hist(bins=15, edgecolor='black', linewidth=1.0,
           xlabelsize=8, ylabelsize=8, grid=False)    
plt.tight_layout() 

# Test set for feature set 1
X_test_1.hist(bins=15, edgecolor='black', linewidth=1.0,
           xlabelsize=8, ylabelsize=8, grid=False)    
plt.tight_layout() 

#%%

# Density Plot
# check that distributions are the same between train and test set for each feature set
# distributions roughly the same

# Feature set 0
fig = plt.figure(figsize=(12,12))
for df in X_featureset_0:
        for i in range(len(df.columns)):
            fig.add_subplot(3,4, i+1)
            sns.kdeplot(df['MFCC_'+str(i)], shade=True)
            plt.xlabel('MFCC_'+str(i))
            plt.legend(['train','test'])

plt.tight_layout()

# Feature set 1
fig = plt.figure(figsize=(12,12))
for df in X_featureset_1:
    for i in range(len(df.columns)):
        fig.add_subplot(4,6, i+1)
        if(i < 12):
            sns.kdeplot(df['MFCC_'+str(i)], shade=True)
            plt.xlabel('MFCC_'+str(i))
        else:
            sns.kdeplot(df['MFCC_var_'+str(i-12)], shade=True)
            plt.xlabel('MFCC_var_'+str(i-12))
        plt.legend(['train','test'])
        
        plt.tight_layout()
        


#%%

#check that target distributions are the same between train and test set

# target distribution roughly the same
for y_data in y:
    sns.distplot(y_data)

    
    
#%%
# Multivariate analysis

# Look for correlation btwn features
for df in X_featureset_0 + X_featureset_1: 
    corrmat = df.corr()
    f, ax = plt.subplots(figsize=(16,12))
    sns.heatmap(corrmat, square=True, annot=True)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    
# MFCC feature 9 & 10 most correlated, around 0.80
    
    
#%%

# Look for correlation w/ target label
x = pd.concat([X_train_0, y_train_0], axis=1)
corrmat = x.corr()
f, ax = plt.subplots(figsize=(16,12))
sns.heatmap(corrmat, square=True, annot=True)
#sns.heatmap(corrmat >0.8, vmax=1.0, square=True, annot=True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)

# MFCC feature 9 slightly more correlated than feature 10 in X_train



#%%
   
# remove MFCC_10 / MFCC_var_10
X_train_0_drop = X_train_0.drop(columns=['MFCC_10'])
X_test_0_drop = X_test_0.drop(columns=['MFCC_10'])

X_train_1_drop = X_train_1.drop(columns=['MFCC_10', 'MFCC_var_10'])
X_test_1_drop = X_test_1.drop(columns=['MFCC_10', 'MFCC_var_10'])

#%%
# write to csv
directory = 'csv'

X_train_0_drop.to_csv(directory + '/' + "X_train_0_drop.csv", index=False)
X_train_1_drop.to_csv(directory + '/' + "X_train_1_drop.csv", index=False)
X_test_0_drop.to_csv(directory + '/' + "X_test_0_drop.csv", index=False)
X_test_1_drop.to_csv(directory + '/' + "X_test_1_drop.csv", index=False)

#%%
# Perform feature scaling with the models in the pipeline