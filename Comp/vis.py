import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
trainDir = './data/TrainingData/'
trainingfiles_all = os.listdir(trainDir)
testfiles_all = os.listdir('./data/TestData')

trainFiles_x = [file for file in trainingfiles_all if "x.csv" in file]
f_id = 1

subjectTraining_all = []

for i in range(len(trainFiles_x)):
    x_file = trainFiles_x[i]
    xt_file = x_file.replace('x.csv', 'x_time.csv')
    y_file = x_file.replace('x.csv', 'y.csv')
    yt_file = x_file.replace('x.csv', 'y_time.csv')
    subjectTraining_all.append([x_file, xt_file, y_file, yt_file])

organizedDF = pd.DataFrame(subjectTraining_all)
#print(organizedDF)
for i, row in organizedDF.iterrows():
    X_data, Xt, y_data, yt = [pd.read_csv(trainDir+file, header=None) for file in row]
    X = pd.concat([X_data, Xt], axis=1)
    X.columns = ["imu1","imu2","imu3","imu4","imu5","imu6","time"]
    y = pd.concat([y_data, yt], axis=1)
    y.columns = ["label", "time"]
    #print(X.describe(), Xt.describe(), y.describe(), yt.describe())
    print(X.describe())
    fig, axes = plt.subplots(nrows=3, ncols=1)
    X.iloc[:,[0,1,2,-1]].plot(x = 'time', ax=axes[0])
    X.iloc[:,[3,4,5,-1]].plot(x = "time", ax=axes[1])
    y.plot(x = "time", ax=axes[2])
    plt.show()
