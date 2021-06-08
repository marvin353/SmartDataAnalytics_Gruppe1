# Resampling using rolling window

import pandas as pd
import os

rootdir_train = '/Users/marvinwoller/Desktop/SmartDataAnalytics/Blatt2/data/train/'
rootdir_test = '/Users/marvinwoller/Desktop/SmartDataAnalytics/Blatt2/data/test/'

train_labels_path = '/Users/marvinwoller/Desktop/SmartDataAnalytics/Blatt2/data/train_label.csv'
test_labels_path = '/Users/marvinwoller/Desktop/SmartDataAnalytics/Blatt2/data/test_label.csv'

# Resampling values (Original samling period: 10 min)
# 1H = 6, 3H = 18, 6H = 36, 12h = 72

count = 0 # This operation may take long time, the count shows process for users satisfaction
dirs = [x[0] for x in os.walk(rootdir_test)]
dirs2skip = [] #Create files in Batches because of Colab timeout
path = "/Users/marvinwoller/Desktop/SmartDataAnalytics/Blatt2/data/resampled_test/resampled_"

times = {
    "1H": 6,
    "3H": 18,
    "6H": 36,
    "12H": 72
}

for subdir in dirs[1:]:
    if subdir in dirs2skip:
        continue
    files = os.listdir(subdir)
    for csv_file in files[0:]:
        df = pd.read_csv(subdir + "/" + csv_file, index_col=0)

        for time in times:

            # perform resampling with rolling window and mean/median
            resampling_value = times[time]
            df_mean = df.rolling(resampling_value).mean()
            df_mean = df_mean.iloc[::resampling_value, :] # Select rows according to rolling window size
            df_mean = df_mean.iloc[1:] # Drop first row, because it contains only NaN values as result of applying rolling window function
            df_median = df.rolling(resampling_value).median()
            df_median = df_median.iloc[::resampling_value, :] # Select rows according to rolling window size
            df_median = df_median.iloc[1:] # Drop first row, because it contains only NaN values as result of applying rolling window function

            # Create Directories and save resampled file as new CSV file
            path_mean = path + time + "/mean/" + subdir[-3:]
            if not os.path.isdir(path_mean):
                try:
                    os.makedirs(path_mean)
                except OSError:
                    print ("Creation of the directory %s failed" % path_mean)
                else:
                    print ("Successfully created the directory %s " % path_mean)

            path_median = path + time + "/median/" + subdir[-3:]
            if not os.path.isdir(path_median):
                try:
                    os.makedirs(path_median)
                except OSError:
                    print ("Creation of the directory %s failed" % path_median)
                else:
                    print ("Successfully created the directory %s " % path_median)

            #Write resample data to csv file
            df_mean.to_csv(path_mean + "/" + csv_file)
            df_median.to_csv(path_median + "/" + csv_file)

        print("Processing file No.: " + str(count))
        count = count + 1
