# Helper file for preprocessing

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import glob
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

train_labels_path = '/Users/marvinwoller/Desktop/SmartDataAnalytics/Blatt2/data/train_label.csv'
test_labels_path = '/Users/marvinwoller/Desktop/SmartDataAnalytics/Blatt2/data/test_label.csv'

train_labels = pd.read_csv(train_labels_path, index_col=0) #Don't use index numbers per row but CSV file name as index
test_labels = pd.read_csv(test_labels_path, index_col=0) #Don't use index numbers per row but CSV file name as index


# --------------------------------- Basic functons -----------------------------------
# Remove Outliers using z-score
def remove_outliers(df_local):
    z = np.abs(stats.zscore(df_local))
    df_local = df_local[(z < 3).all(axis=1)]
    return df_local

# Perform Scaling -> Scale to intervall [0,1]
scaler = MinMaxScaler()
def perform_scaling(df_local):
    df_local[df_local.columns] = scaler.fit_transform(df_local[df_local.columns])
    return df_local

def remove_unnecessaray_columns(df_local):
    df_local.drop('Drive 1 output torque', inplace=True, axis=1)
    df_local.drop('Drive 2 output torque', inplace=True, axis=1)
    df_local.drop('Drive 3 output torque', inplace=True, axis=1)
    return df_local

def getLabel_train(name):
    rowData = train_labels.loc[ name , : ]
    return rowData['ret']

def getLabel_test(name):
    rowData = test_labels.loc[ name , : ]
    return rowData['ret']

# -------------------------------- Special functions ---------------------------------
def basic_preprocessing(df_local):
    #df_local = remove_unnecessaray_columns(df_local)
    df_local = remove_outliers(df_local)
    df_local = perform_scaling(df_local)
    return df_local

def preprocessing_SVM_12H_mean(df_local):
    df_local = remove_unnecessaray_columns(df_local)
    df_local = remove_outliers(df_local)
    return df_local