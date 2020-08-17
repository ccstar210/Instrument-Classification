# Extract the MFCC features from the data.csv file.

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import helper # contains the helper functions


#%%
# Load the raw data
dataFile = "data.csv"
raw_data = pd.read_csv(dataFile, header=0)

raw_data.info()

# no missing values
raw_data.isnull().sum()

X = raw_data.iloc[:,0] # name of recordings
y = raw_data.iloc[:,1] # instrument label (0-10)


#%%
#Preliminary EDA

print("Describe: \n")
print(y.describe())

print("Dtypes: \n")
print(y.dtypes)

# total instrument distribution
sns.distplot(y)
print("skewness: %f" % y.skew())
print("Kurtosis: %f" % y.kurt())

#%%

# Separate the training data into training and validation set
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

#%%
# Extract features (MFCC's)

# Investigating 2 types of feature sets ((0) mean MFCCs and (1) mean MFCCs + variance)
numFeatureSettings = 2

print('Extracting MFCC features...')
for i in range(numFeatureSettings):

        
    X_trn_data, y_trn_data = helper.formatData(X_trn, y_trn, i)
    X_tst_data, y_tst_data = helper.formatData(X_tst, y_tst, i)
    
        
    # Write data to the csv file and rename column titles
    if i == 0:
        pd.DataFrame(X_trn_data).to_csv("X_train_"+str(i), index=False, header=["MFCC_0", "MFCC_1", "MFCC_2", "MFCC_3", "MFCC_4", "MFCC_5", "MFCC_6",
                          "MFCC_7", "MFCC_8", "MFCC_9", "MFCC_10", "MFCC_11"])
        pd.DataFrame(X_tst_data).to_csv("X_test_"+str(i), index=False, header=["MFCC_0", "MFCC_1", "MFCC_2", "MFCC_3", "MFCC_4", "MFCC_5", "MFCC_6",
                          "MFCC_7", "MFCC_8", "MFCC_9", "MFCC_10", "MFCC_11"])
    else:
        pd.DataFrame(X_trn_data).to_csv("X_train_"+str(i), index=False, header=["MFCC_0", "MFCC_1", "MFCC_2", "MFCC_3", "MFCC_4", "MFCC_5", "MFCC_6",
                          "MFCC_7", "MFCC_8", "MFCC_9", "MFCC_10", "MFCC_11","MFCC_var_0", "MFCC_var_1", "MFCC_var_2", "MFCC_var_3", "MFCC_var_4", 
                          "MFCC_var_5", "MFCC_var_6","MFCC_var_7", "MFCC_var_8", "MFCC_var_9","MFCC_var_10","MFCC_var_11"])
        pd.DataFrame(X_tst_data).to_csv("X_test_"+str(i), index=False, header=["MFCC_0", "MFCC_1", "MFCC_2", "MFCC_3", "MFCC_4", "MFCC_5", "MFCC_6",
                          "MFCC_7", "MFCC_8", "MFCC_9", "MFCC_10", "MFCC_11", "MFCC_var_0", "MFCC_var_1", "MFCC_var_2", "MFCC_var_3", "MFCC_var_4", 
                          "MFCC_var_5", "MFCC_var_6","MFCC_var_7", "MFCC_var_8", "MFCC_var_9","MFCC_var_10","MFCC_var_11"])
     
    pd.DataFrame(y_trn_data).to_csv("y_train_"+str(i), index=False, header=['label'])
    pd.DataFrame(y_tst_data).to_csv("y_test_"+str(i), index=False, header=['label'])
    

print('Feature extraction complete')

    



