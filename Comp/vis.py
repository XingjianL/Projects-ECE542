import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import List
import random
import seaborn as sns
def getDataset(dir:str):
    files_all = os.listdir(dir)

    Files_x = [file for file in files_all if "x.csv" in file]

    subjectTraining_all = []

    for i in range(len(Files_x)):
        x_file = Files_x[i]
        xt_file = x_file.replace('x.csv', 'x_time.csv')
        y_file = x_file.replace('x.csv', 'y.csv')
        yt_file = x_file.replace('x.csv', 'y_time.csv')
        subjectTraining_all.append([x_file, xt_file, y_file, yt_file])

    organizedDF = pd.DataFrame(subjectTraining_all)
    organizedX = []
    organizedY = []
    #print(organizedDF)
    for i, row in organizedDF.iterrows():
        X_data, Xt, y_data, yt = [pd.read_csv(dir+file, header=None) for file in row]
        X = pd.concat([X_data, Xt], axis=1)
        X.columns = ["imu1","imu2","imu3","imu4","imu5","imu6","time"]
        y = pd.concat([y_data, yt], axis=1)
        y.columns = ["label", "time"]
        organizedX.append(X)
        organizedY.append(y)
        #print(X.describe(), Xt.describe(), y.describe(), yt.describe())
        #print(X.describe())
        #fig, axes = plt.subplots(nrows=3, ncols=1)
        #X.iloc[:,[0,1,2,-1]].plot(x = 'time', ax=axes[0])
        #X.iloc[:,[3,4,5,-1]].plot(x = "time", ax=axes[1])
        #y.plot(x = "time", ax=axes[2])
        #plt.show()
    #print(f"Dataset obtained: \n{X.describe()}\n{y.describe()}")
    return organizedX, organizedY

def splitTrainVal(train_set_x, train_set_y, val_train_split = 1/10):
    total_samples = sum([len(x) for x in train_set_x])
    val_idxs = [np.random.randint(0,len(train_set_x))]
    
    val_samples = sum([len(train_set_x[idx]) for idx in val_idxs])
    while val_samples/total_samples < val_train_split:
        new_sample = np.random.choice([x for x in range(len(train_set_x)) if x not in val_idxs])
        val_idxs.append(new_sample)
        val_samples = sum([len(train_set_x[idx]) for idx in val_idxs])
    #print(val_samples)
    #print(val_samples/total_samples)
    
    train_idxs = [x for x in range(len(train_set_x)) if x not in val_idxs]
    print(train_idxs)
    print(val_idxs)
    return train_idxs, val_idxs

def combineAndUpsample(X : List[pd.DataFrame], y : List[pd.DataFrame], consistent_times = True):
    #print(len(X))
    #print(len(y))
    all_upsampled = []
    for i in range(len(y)):
        X_i = X[i]
        y_i = y[i]
        #print(X_i)
        #print(y_i)
        combined = pd.merge(X_i, y_i, how='outer', on='time', sort='True')
        upsampled = combined.interpolate(method='nearest')
        
        upsampled['label'] = upsampled['label'].fillna(0)
        upsampled['label'] = upsampled['label'].astype(int)

        if consistent_times:
            rows_to_remove = upsampled[((((upsampled['time'] / 0.025) % 1) > 0.001) & ((upsampled['time'] % 0.025) < 0.024))].index
            #print(rows_to_remove)
            #print(0.075 % 0.025)
            upsampled.drop(rows_to_remove, inplace=True)
        
        all_upsampled.append(upsampled)

        #print(upsampled)
    return all_upsampled

def oneHotEncode(dataset_list):
    for dataset in dataset_list:
        dataset["label_0"] = (dataset["label"] == 0).astype(int)
        dataset["label_1"] = (dataset["label"] == 1).astype(int)
        dataset["label_2"] = (dataset["label"] == 2).astype(int)
        dataset["label_3"] = (dataset["label"] == 3).astype(int)

    return dataset_list


def prepareData(filepath : str, plot = False, consistent_times = True, one_hot_encoding = True, val_split = 1/10):
    # list of X dataframe each with [imu1-6, time], list of y dataframe each with [label, time]
    X, y = getDataset(filepath) 
    
    train_idxs, val_idxs = splitTrainVal(X, y, val_train_split=val_split)
    val_set_x = [X[i] for i in val_idxs]
    val_set_y = [y[i] for i in val_idxs]
    train_set_x = [X[i] for i in train_idxs]
    train_set_y = [y[i] for i in train_idxs]

    # combine X and y and upsample and interpolate to X_time
    training_set = combineAndUpsample(train_set_x, train_set_y, consistent_times=consistent_times)
    validation_set = combineAndUpsample(val_set_x, val_set_y, consistent_times=consistent_times)

    if plot:
        fig, axes = plt.subplots(nrows=3, ncols=2)
        axes[0,0].title.set_text("Interpolated Data \n(linear, should be same as original plot)")
        axes[0,1].title.set_text("Original Data")
        train_set_x[0].iloc[:,[0,1,2,-1]].plot(x = 'time', ax=axes[0,1])
        train_set_x[0].iloc[:,[3,4,5,-1]].plot(x = 'time', ax=axes[1,1])
        train_set_y[0].plot(x = 'time', ax=axes[2,1])
        training_set[0].iloc[:,[0,1,2,-2]].plot(x = 'time', ax=axes[0,0])
        training_set[0].iloc[:,[3,4,5,-2]].plot(x = 'time', ax=axes[1,0])
        training_set[0].iloc[:,[-1,-2]].plot(x = 'time', ax=axes[2,0])
        plt.show()

    if one_hot_encoding:
        training_set = oneHotEncode(training_set)
        validation_set = oneHotEncode(validation_set)
    print(f"Dataset obtained: training {len(training_set)}, val {len(validation_set)}. \
          \n datapoints: train {sum([len(x) for x in training_set])}, val {sum([len(x) for x in validation_set])}.")
    #print(f"Example (training_set[0]):\n{training_set[0].head(3)}\n")
    return training_set, validation_set

def prepareBatchedData(datasets, interval = 80, batch_freq = 1, randomize = True, noise = True, seq_output = True, force_all = False):
    batched_data_X = []
    batched_data_y = []
    #print(f"\t{randomize}")
    for dataset in datasets:
        np_data = dataset.to_numpy().astype(np.float32)
        # how many batches from this dataset
        batch_count = (np.floor(len(dataset)/interval)).astype(int)
        if randomize:
            batch_count = batch_count * batch_freq
            batch_idx = random.sample(range(0,len(dataset)-interval), batch_count)
        else:
            batch_idx = [interval*i for i in range(batch_count)]
        if seq_output is False:
            # reduce num of batches, otherwise it would be too much to train (in the millions)
            # evenly sampled, most data points appear batch_freq times in the dataset
            batch_idx = [i*int(interval/batch_freq) for i in range(int(len(dataset)/int(interval/batch_freq)))]
            if force_all:
                batch_idx = [i for i in range(int(len(dataset)))]
            # insert padding of zeros ahead
            np_data = np.insert(np_data,0,np.zeros((interval, 6+1+1+4)), axis=0) # 6 imu, 1 time, 1 label, 4 one hot columns

        X = []
        y = []
        for i in batch_idx:
            xdata = np_data[i:i+interval,:6].reshape([1,interval,6])
            if noise:
                e = np.random.normal(0, 5, size = [1,interval,6])
                xdata = xdata + e
                # zscore norm
            mean = np.mean(xdata, axis=1)
            std = np.std(xdata, axis=1)
            xdata = np.nan_to_num((xdata - mean)/std, nan=0)
                # unit norm
            #norm = np.linalg.norm(xdata, axis=1)
            #xdata = np.nan_to_num(xdata / norm,nan=0)
            #print(mean.shape)
            #print(std.shape)
            X.append(xdata)
            if seq_output:
                y.append(np_data[i:i+interval,8:].reshape([1,interval,4]))
            else:
                y.append(np_data[i+interval-1,8:].reshape([1,1,4]))
        
        batched_data_X += X
        batched_data_y += y
    print(f"num_batches: {len(batched_data_X)}, each X batch: {X[0].shape}, each Y batch: {y[0].shape}")
    batched_data_X = np.array(batched_data_X).reshape((len(batched_data_X), interval, 6))
    if seq_output:
        batched_data_y = np.array(batched_data_y).reshape((len(batched_data_y), interval, 4))
    else:
        batched_data_y = np.array(batched_data_y).reshape((len(batched_data_y), 1, 4))
    
    return batched_data_X, batched_data_y

def frequencyDomain(data, only_fft = True):
    freq = np.fft.fft(data,data.shape[1], axis=1)
    print(freq[0].shape)
    if only_fft:
        return freq
    combined = np.concatenate((data, freq), axis=2)
    print(combined.shape)
    return combined

if __name__ == "__main__":
    train_dir = '/home/xing/Classes/ECE542/Project/Projects-ECE542/Comp/data/TrainingData/'
    train_dir = '/home/lixin/Classes/Spr23/542/Projects-ECE542/Comp/data/TrainingData/'

    training_set, validation_set = prepareData(train_dir, plot=False)
    train_set_x, train_set_y = prepareBatchedData(training_set)
    train_set_x = frequencyDomain(train_set_x, only_fft=False)

    sns.set()
    fig, axes = plt.subplots(3,3)
    for i in range(3):
        for j in range(3):
            axes[i,j].plot(train_set_x[100*(i*3+j), :, :],label="IMU")
            #plt.plot(train_set_x[0, :, 1], label="Input feature 1")
            axes[i,j].plot(100*train_set_y[100*(i*3+j), :, :], label="Label")
            #print(train_set_y[100*(i*3+j), :, :])
            #axes[i,j].ylim((-1.1, 1.1))
            #axes[i,j].title("Training data")
            #axes[i,j].legend(loc="upper right")
    plt.show()
    #test_dir = './data/TestData/'
    #test_set = getDataset(test_dir)
