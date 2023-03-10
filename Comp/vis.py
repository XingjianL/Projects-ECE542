import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
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

def splitTrainVal(train_set_x, train_set_y, val_train_split = 1/5):
    total_samples = sum([len(x) for x in train_set_x])
    val_idxs = [np.random.randint(0,len(train_set_x))]
    
    val_samples = sum([len(train_set_x[idx]) for idx in val_idxs])
    while val_samples/total_samples < val_train_split:
        new_sample = np.random.choice([x for x in range(len(train_set_x)) if x not in val_idxs])
        val_idxs.append(new_sample)
        val_samples = sum([len(train_set_x[idx]) for idx in val_idxs])
    print(val_samples)
    print(val_samples/total_samples)
    print(val_idxs)
    train_idxs = [x for x in range(len(train_set_x)) if x not in val_idxs]
    print(train_idxs)
    return train_idxs, val_idxs


if __name__ == "__main__":
    train_dir = '/home/lixin/Classes/Spr23/542/Projects-ECE542/Comp/data/TrainingData/'
    train_set_x, train_set_y = getDataset(train_dir)
    print(len(train_set_x))
    total_samples = sum([len(x) for x in train_set_x])
    print(total_samples)
    train_idxs, val_idxs = splitTrainVal(train_set_x, train_set_y)
    #test_dir = './data/TestData/'
    #test_set = getDataset(test_dir)
