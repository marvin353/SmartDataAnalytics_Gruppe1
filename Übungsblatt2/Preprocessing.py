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
    df_local.drop('Drive 1 output torque', inplace=True, axis=1, errors='ignore')
    df_local.drop('Drive 2 output torque', inplace=True, axis=1, errors='ignore')
    df_local.drop('Drive 3 output torque', inplace=True, axis=1, errors='ignore')
    return df_local

def drop_columns_with_drift(df_local, intensity='strong'):
    drifting_sensors_strong = ['pitch motor 1 current', 'pitch motor 2 current', 'Pitch motor 3 current', 'x direction vibration value', 'y direction vibration value', 'hydraulic brake pressure', 'generator current', 'Inverter inlet temperature', 'inverter outlet temperature', 'inverter inlet pressure', 'inverter outlet pressure', 'wind tower ambient temperature', 'Wheel temperature', 'Wheel control cabinet temperature', 'Cabin temperature', 'Cabin control cabinet temperature', 'vane 1 pitch motor temperature', 'blade 2 pitch motor temperature', 'blade 3 pitch motor temperature', 'blade 1 inverter box temperature', 'blade 2 inverter box temperature', 'blade 3 inverter box temperature', 'blade 1 super capacitor voltage', 'blade 2 super capacitor voltage', 'blade 3 super capacitor voltage', 'drive 1 thyristor temperature', 'Drive 2 thyristor temperature', 'Drive 3 thyristor temperature']
    drifting_sensors_medium = ['inverter grid side current', 'Inverter grid side active power', 'inverter generator side power', 'generator operating frequency','generator stator temperature 1', 'generator stator temperature 2', 'generator stator temperature 3', 'generator stator temperature 4', 'Generator stator temperature 5', 'generator stator temperature 6', 'generator air temperature 1', 'generator air temperature 2', 'main bearing temperature 1', 'main bearing temperature 2', 'Pitch motor 1 power estimation', 'Pitch motor 2 power estimation', 'Pitch motor 3 power estimation', 'blade 1 battery box temperature', 'blade 2 battery box temperature', 'blade 3 battery box temperature']
    drifting_sensors_light = ['Inverter INU temperature', 'Inverter ISU temperature']

    if intensity == 'strong':
        df_local = df_local.drop(drifting_sensors_strong, axis=1, errors='ignore')
    elif intensity == 'medium':
        df_local = df_local.drop(drifting_sensors_strong, axis=1, errors='ignore')
        df_local = df_local.drop(drifting_sensors_medium, axis=1, errors='ignore')
    elif intensity == 'light':
        df_local = df_local.drop(drifting_sensors_strong, axis=1, errors='ignore')
        df_local = df_local.drop(drifting_sensors_medium, axis=1, errors='ignore')
        df_local = df_local.drop(drifting_sensors_light, axis=1, errors='ignore')

    return df_local

def remove_columns_with_correlation(df_local):
    cols2drop_correlation = ['blade 1 angle', 'blade 2 angle', 'blade 3 angle', 'pitch motor 1 current', 'pitch motor 2 current', 'Pitch motor 3 current', 'generator stator temperature 1', 'generator stator temperature 2', 'generator stator temperature 3', 'generator stator temperature 4', 'Generator stator temperature 5', 'generator stator temperature 6', 'Pitch motor 1 power estimation', 'Pitch motor 2 power estimation', 'Pitch motor 3 power estimation', 'blade 2 battery box temperature', 'blade 3 battery box temperature', 'blade 2 pitch motor temperature', 'blade 3 pitch motor temperature', 'blade 1 inverter box temperature', 'blade 2 inverter box temperature', 'blade 1 super capacitor voltage', 'blade 2 super capacitor voltage', 'blade 3 super capacitor voltage', 'drive 1 thyristor temperature', 'Drive 2 thyristor temperature', 'Drive 3 thyristor temperature']
    try:
        df_local['blade angle']                        = df_local.loc[: , 'blade 1 angle':'blade 3 angle'].median(axis=1)
    except KeyError:
        print("KeyError skipping...")
    try:
        df_local['pitch motor current']                = df_local.loc[: , 'pitch motor 1 current':'Pitch motor 3 current'].median(axis=1)
    except KeyError:
        print("KeyError skipping...")
    try:
        df_local['generator stator temperature']       = df_local.loc[: , 'generator stator temperature 1':'generator stator temperature 6'].median(axis=1)
    except KeyError:
        print("KeyError skipping...")
    try:
        df_local['Pitch motor power estimation']       = df_local.loc[: , 'Pitch motor 1 power estimation':'Pitch motor 3 power estimation'].median(axis=1)
    except KeyError:
        print("KeyError skipping...")
    try:
        df_local['blade 2 3 battery box temperature']  = df_local.loc[: , 'blade 2 battery box temperature':'blade 3 battery box temperature'].median(axis=1)
    except KeyError:
        print("KeyError skipping...")
    try:
        df_local['blade 2 3 pitch motor temperature']  = df_local.loc[: , 'blade 2 pitch motor temperature':'blade 3 pitch motor temperature'].median(axis=1)
    except KeyError:
        print("KeyError skipping...")
    try:
        df_local['blade 1 2 inverter box temperature'] = df_local.loc[: , 'blade 1 inverter box temperature':'blade 2 inverter box temperature'].median(axis=1)
    except KeyError:
        print("KeyError skipping...")
    try:
        df_local['blade super capacitor voltage']      = df_local.loc[: , 'blade 1 super capacitor voltage':'blade 3 super capacitor voltage'].median(axis=1)
    except KeyError:
        print("KeyError skipping...")
    try:
        df_local['drive thyristor temperature']        = df_local.loc[: , 'drive 1 thyristor temperature':'Drive 3 thyristor temperature'].median(axis=1)
    except KeyError:
        print("KeyError skipping...")

    df_local = df_local.drop(cols2drop_correlation, axis=1, errors='ignore')

    return df_local

def get_good_sensors(df_local):
    # The good sensors list is a result of the analysis of the importance of features using only one area (064)
    good_sensors = ['hub angle', 'blade 1 angle', 'blade 2 angle', 'blade 3 angle', 'hydraulic brake pressure', 'Aircraft weather station wind speed', 'wind direction absolute value', 'inverter grid side current', 'inverter grid side voltage', 'Inverter grid side active power', 'inverter generator side power', 'generator current', 'generator torque', 'generator power limit value', 'Rated hub speed', 'wind tower ambient temperature', 'generator stator temperature 1', 'generator stator temperature 2', 'generator stator temperature 3', 'generator stator temperature 4', 'Generator stator temperature 5', 'generator stator temperature 6', 'generator air temperature 1', 'generator air temperature 2', 'main bearing temperature 1', 'main bearing temperature 2', 'Wheel temperature', 'Wheel control cabinet temperature', 'Cabin temperature', 'Cabin control cabinet temperature', 'blade 1 battery box temperature', 'blade 2 battery box temperature', 'blade 3 battery box temperature', 'blade 1 inverter box temperature', 'blade 2 inverter box temperature', 'blade 3 inverter box temperature', 'blade 1 super capacitor voltage', 'blade 2 super capacitor voltage', 'blade 3 super capacitor voltage', 'drive 1 thyristor temperature', 'Drive 2 thyristor temperature', 'Drive 3 thyristor temperature']
    df_good_sensors = df_local[good_sensors]
    return df_good_sensors


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

def preprocess(df_local, random_n = 3000, rem_outlier=False, rem_drift=True, rem_corr=False, get_good=False, scale=True):
    if rem_outlier:
        df_local = remove_outliers(df_local)
    df_local = df_local.sample(n=random_n, random_state=123)
    labels = df_local['label']
    df_local.drop(['area','label'], axis = 1, inplace=True)
    if get_good:
        df_local = get_good_sensors(df_local)
    if rem_drift:
        df_local = drop_columns_with_drift(df_local, intensity='strong')
    if rem_corr:
        df_local = remove_columns_with_correlation(df_local)
    if scale:
        df_local = perform_scaling(df_local)
    return labels.values, df_local.values

def preprocess_test(df_local, rem_outlier=False, rem_drift=True, rem_corr=False, get_good=False, scale=True):
    df_local= df_local.dropna() # Drop rows with missing label
    if rem_outlier:
        df_local = remove_outliers(df_local)
    labels_test = df_local['label']
    df_local.drop(['area','label'], axis = 1, inplace=True)
    if get_good:
        df_local = get_good_sensors(df_local)
    if rem_drift:
        df_local = drop_columns_with_drift(df_local, intensity='strong')
    if rem_corr:
        df_local = remove_columns_with_correlation(df_local)
    if scale:
        df_local = perform_scaling(df_local)
    return labels_test.values, df_local.values