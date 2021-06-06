import numpy as np
import pandas as pd
import sklearn
import glob
import Preprocessing as pp

""" This file contains any kind of helper functions """

# Generate numpy arrays with data and labels from all CSV files by iterating over all subfolders in the directory with path 'path' (This works only with original or resampled timeseries data)
def data_label_generator_for_resampled_data(path, train_or_test='train'):
    path_len = len(path) + 4 # Path length + 046/ for example
    files = glob.glob(path + '**/*.csv')
    len_files = len(files)
    count = 0
    data = []
    labels = []
    printProgressBar(0, len_files, prefix='Reading CSVs:', suffix='Complete', length=50, printEnd='') # Initial call
    for file in files[0:]:
        count = count + 1
        name = file[path_len:]
        printProgressBar(count, len_files, prefix='Reading CSVs:', suffix='Complete', length=50, printEnd='')
        temp = pd.read_csv(file) # Change this line to read any other type of file
        pp.preprocessing_SVM_12H_mean(temp)
        if train_or_test == 'train':
            label = pp.getLabel_train(name)
        else:
            label = pp.getLabel_test(name)
        for row in temp.values:
            if label == 1.0 or label == 0.0:
                data.append(row)
                labels.append(label)
    data = np.asarray(data)
    labels = np.asarray(labels)
    return data,labels


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


# Helper function to read data in batches
def data_generator(file_list, dir_path, batch_size = 20):
    path_len = len(dir_path) + 4 #path length + 046/ for example
    i = 0
    while True:
        if i*batch_size >= len(file_list):  # This loop is used to run the generator indefinitely.
            i = 0
            np.random.shuffle(file_list)
        else:
            file_chunk = file_list[i*batch_size:(i+1)*batch_size]
            data = []
            labels = []
            for file in file_chunk:
                temp = pd.read_csv(open(file,'r')) # Change this line to read any other type of file
                data.append(temp.values)
                name = file[path_len:]
                label = pp.getLabel_train(name)
                for i in range(len(temp)):
                    labels.append(label)
            data = np.asarray(data)
            labels = np.asarray(labels)
            yield data, labels
            i = i + 1