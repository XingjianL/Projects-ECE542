import torch
import torch.nn as nn
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

import model
import vis
_config_GRU = {
    "val_split" : 2/10,
    "lr" : 0.0001,
    "hidden" : 36,
    "num_stack_cells" : 2,
    "interval" : 60,
    "batch_freq" : 6,
    "epochs" : 100,
    "seq_out" : False,
    "batch_size" : 128,
    "freq" : False,
    "freq_only" : False,
    "TBPTT" : False  # truncated back prop through time (not useful if the batches are shuffled anyways, or no longer in timeseries between batches) https://datascience.stackexchange.com/questions/118030/why-are-the-hidden-states-of-an-rnn-initialised-every-epoch-instead-of-every-bat
}
def test_loader(directory, multi_to_one = True):
    files_all = os.listdir(directory)

    Files_x = [file for file in files_all if "x.csv" in file]

    subjectTraining_all = []

    for i in range(len(Files_x)):
        x_file = Files_x[i]
        xt_file = x_file.replace('x.csv', 'x_time.csv')
        yt_file = x_file.replace('x.csv', 'y_time.csv')
        subjectTraining_all.append([x_file, xt_file, yt_file])

    organizedDF = pd.DataFrame(subjectTraining_all)
    organizedX = []
    organizedY = []
    #print(organizedDF)
    prepend_empty = pd.DataFrame(np.zeros((79,7)))
    prepend_empty.columns = ["imu1","imu2","imu3","imu4","imu5","imu6","time"]
    for i, row in organizedDF.iterrows():
        X_data, Xt, y = [pd.read_csv(directory+file, header=None) for file in row]
        X = pd.concat([X_data, Xt], axis=1)
        X.columns = ["imu1","imu2","imu3","imu4","imu5","imu6","time"]
        if multi_to_one:
            X = pd.concat([prepend_empty, X], axis=0)
        X["pred"] = -1
        y.columns = ["time"]
        y['pred'] = -1
        y = y.reindex(columns=["label", "time"])
        organizedX.append(X)
        organizedY.append(y)

    return organizedX, organizedY

if __name__ == "__main__":
    testDir = "/home/xing/Classes/ECE542/Project/Projects-ECE542/Comp/data/TestData/"
    model_path = ""
    X_list, y_list = test_loader(testDir)

    for X_df in X_list:
        grumodel = model.GRUModel(6,hidden_size=_config_GRU["hidden"],num_layers=_config_GRU["num_stack_cells"],output_size=4, all_output=_config_GRU["seq_out"]).cuda()
        #print(X)
        X = X_df.to_numpy()

        for i in range(len(X)):
            if not _config_GRU["TBPTT"]:
                h0 = None
                
            batch = torch.from_numpy(X[i:i+_config_GRU["interval"]][:,:6]).view(-1, _config_GRU["interval"], 6).float().cuda() # single batch
            outputs, h1 = grumodel(batch, h0)
            pred = torch.argmax(outputs, dim=1)
            print(i,(batch.shape))