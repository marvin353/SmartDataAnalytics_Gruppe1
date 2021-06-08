# Calculate mean,median,... for each full timeseries (= for each column of csv file) and concatenate all mean,median,... values of all csv files. Also append new column with area identifier and label
# The resulting file would then look like this (for train data):
# AreaCode, Feature 1 ... Feature n, Label
#                  ...
#               33360 rows

import pandas as pd
import os
import glob
import Preprocessing as pp
import Helpers

rootdir_train = '/Users/marvinwoller/Desktop/SmartDataAnalytics/Blatt2/data/train/'
rootdir_test = '/Users/marvinwoller/Desktop/SmartDataAnalytics/Blatt2/data/test/'

train_labels_path = '/Users/marvinwoller/Desktop/SmartDataAnalytics/Blatt2/data/train_label.csv'
test_labels_path = '/Users/marvinwoller/Desktop/SmartDataAnalytics/Blatt2/data/test_label.csv'

path2write = '/Users/marvinwoller/Desktop/SmartDataAnalytics/Blatt2/data/features/'


dirs = [x[0] for x in os.walk(rootdir_train)]
count = 0
files = glob.glob(rootdir_train + '**/*.csv')
len_files = len(files)

main_df_mean = pd.DataFrame()
main_df_median = pd.DataFrame()
main_df_min = pd.DataFrame()
main_df_max = pd.DataFrame()
main_df_std = pd.DataFrame()
main_df_var = pd.DataFrame()

for subdir in dirs[1:]:
    files = os.listdir(subdir)
    for csv_file in files[0:]:
        # Show count because its more user friendly
        #clear_output(wait=False)
        #print("Process file %i " % count)
        Helpers.printProgressBar(count, len_files, prefix='Processing files:', suffix='Complete', length=50, printEnd='')
        count = count + 1

        # Processing
        df = pd.read_csv(subdir + "/" + csv_file)
        df = pp.remove_unnecessaray_columns(df) # Remove all columns that contain no information
        area = subdir[-3:] # Extract the area code from path
        label = pp.getLabel_train(csv_file) # Get the label by file name. !!! IMPORTANT: For training data use getlabel_train(), for test data getLabel_test() !!!

        df_mean = df.mean(axis=0).to_frame().T
        df_mean['area'] = area # Add area code and label to dataframe
        df_mean['label'] = label

        df_median = df.median(axis=0).to_frame().T # Find label for CSV file
        df_median['area'] = area # Add area code and label to dataframe
        df_median['label'] = label

        df_min = df.min(axis=0).to_frame().T # Find label for CSV file
        df_min['area'] = area # Add area code and label to dataframe
        df_min['label'] = label

        df_max = df.max(axis=0).to_frame().T # Find label for CSV file
        df_max['area'] = area # Add area code and label to dataframe
        df_max['label'] = label

        df_std = df.std(axis=0).to_frame().T # Find label for CSV file
        df_std['area'] = area # Add area code and label to dataframe
        df_std['label'] = label

        df_var = df.var(axis=0).to_frame().T # Find label for CSV file
        df_var['area'] = area # Add area code and label to dataframe
        df_var['label'] = label

        # Calculate Features over full time series
        main_df_mean = main_df_mean.append(df_mean)
        main_df_median = main_df_median.append(df_median)
        main_df_min = main_df_min.append(df_min)
        main_df_max = main_df_max.append(df_max)
        main_df_std = main_df_std.append(df_std)
        main_df_var = main_df_var.append(df_var)

main_df_mean = main_df_mean.reset_index()
main_df_mean.drop('index', inplace=True, axis=1)

main_df_median = main_df_median.reset_index()
main_df_median.drop('index', inplace=True, axis=1)

main_df_min = main_df_min.reset_index()
main_df_min.drop('index', inplace=True, axis=1)

main_df_max = main_df_max.reset_index()
main_df_max.drop('index', inplace=True, axis=1)

main_df_std = main_df_std.reset_index()
main_df_std.drop('index', inplace=True, axis=1)

main_df_var = main_df_var.reset_index()
main_df_var.drop('index', inplace=True, axis=1)

# Write to new CSV files
main_df_mean.to_csv(path2write + "mean.csv")
main_df_median.to_csv(path2write + "median.csv")
main_df_min.to_csv(path2write + "min.csv")
main_df_max.to_csv(path2write + "max.csv")
main_df_std.to_csv(path2write + "std.csv")
main_df_var.to_csv(path2write + "var.csv")