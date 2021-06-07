import glob
import Helpers
import Preprocessing as pp
import pandas as pd
import numpy as np

path12median = '/Users/marvinwoller/Desktop/SmartDataAnalytics/Blatt2/data/resampled/resampled_12H/median/'
path12mean = '/Users/marvinwoller/Desktop/SmartDataAnalytics/Blatt2/data/resampled/resampled_12H/mean/'
path12median_write = '/Users/marvinwoller/Desktop/SmartDataAnalytics/Blatt2/data/resampled/resampled_12H_median.csv'
path12mean_write = '/Users/marvinwoller/Desktop/SmartDataAnalytics/Blatt2/data/resampled/resampled_12H_mean.csv'

path6median = '/Users/marvinwoller/Desktop/SmartDataAnalytics/Blatt2/data/resampled/resampled_6H/median/'
path6mean = '/Users/marvinwoller/Desktop/SmartDataAnalytics/Blatt2/data/resampled/resampled_6H/mean/'
path6median_write = '/Users/marvinwoller/Desktop/SmartDataAnalytics/Blatt2/data/resampled/resampled_6H_median.csv'
path6mean_write = '/Users/marvinwoller/Desktop/SmartDataAnalytics/Blatt2/data/resampled/resampled_6H_mean.csv'

path3median = '/Users/marvinwoller/Desktop/SmartDataAnalytics/Blatt2/data/resampled/resampled_3H/median/'
path3mean = '/Users/marvinwoller/Desktop/SmartDataAnalytics/Blatt2/data/resampled/resampled_3H/mean/'
path3median_write = '/Users/marvinwoller/Desktop/SmartDataAnalytics/Blatt2/data/resampled/resampled_3H_median.csv'
path3mean_write = '/Users/marvinwoller/Desktop/SmartDataAnalytics/Blatt2/data/resampled/resampled_3H_mean.csv'

path1median = '/Users/marvinwoller/Desktop/SmartDataAnalytics/Blatt2/data/resampled/resampled_1H/median/'
path1mean = '/Users/marvinwoller/Desktop/SmartDataAnalytics/Blatt2/data/resampled/resampled_1H/mean/'
path1median_write = '/Users/marvinwoller/Desktop/SmartDataAnalytics/Blatt2/data/resampled/resampled_1H_median.csv'
path1mean_write = '/Users/marvinwoller/Desktop/SmartDataAnalytics/Blatt2/data/resampled/resampled_1H_mean.csv'

def combine(path, train_or_test='train'):
    path_len = len(path) + 4 # Path length + 046/ for example
    files = glob.glob(path + '**/*.csv')
    len_files = len(files)
    count = 0
    data = pd.DataFrame()
    Helpers.printProgressBar(0, len_files, prefix='Reading CSVs:', suffix='Complete', length=50, printEnd='') # Initial call
    for file in files[0:]:
        labels = []
        areas = []
        count = count + 1
        name = file[path_len:]
        area = file[path_len:path_len+3]
        Helpers.printProgressBar(count, len_files, prefix='Reading CSVs:', suffix='Complete', length=50, printEnd='')
        temp = pd.read_csv(file) # Change this line to read any other type of file
        #pp.preprocessing_SVM_12H_mean(temp)
        if train_or_test == 'train':
            label = pp.getLabel_train(name)
        else:
            label = pp.getLabel_test(name)
        for row in temp.values:
            if label == 1.0 or label == 0.0:
                #data.append(row)
                labels.append(label)
                areas.append(area)
        temp['label'] = labels
        temp['area'] = areas
    data = data.append(temp)
    return data

data = combine(path12median)
data.to_csv(path12median_write)
data = combine(path12mean)
data.to_csv(path12mean_write)

data = combine(path6median)
data.to_csv(path6median_write)
data = combine(path6mean)
data.to_csv(path6mean_write)

data = combine(path3median)
data.to_csv(path3median_write)
data = combine(path3mean)
data.to_csv(path3mean_write)

data = combine(path1median)
data.to_csv(path1median_write)
data = combine(path1mean)
data.to_csv(path1mean_write)