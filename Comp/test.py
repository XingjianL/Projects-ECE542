import torch
import torch.nn as nn
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

import model
import vis

from scipy import stats

# copy the config from train_gru
_config_GRU = {
    "val_split" : 2/10,
    "lr" : 0.001,
    "min_lr" : 0.0001,
    "hidden" : 24,
    "num_stack_cells" : 3,
    "interval" : 60,
    "batch_freq" : 6,
    "epochs" : 100,
    "seq_out" : False,
    "batch_size" : 128,
    "freq" : False,
    "freq_only" : False,
    "TBPTT" : False  # truncated back prop through time (not useful if the batches are shuffled anyways, or no longer in timeseries between batches) https://datascience.stackexchange.com/questions/118030/why-are-the-hidden-states-of-an-rnn-initialised-every-epoch-instead-of-every-bat
}
_config_GRU = {
    "val_split" : 2/10,
    "lr" : 0.001,
    "min_lr" : 0.0001,
    "hidden" : 36,
    "num_stack_cells" : 1,
    "interval" : 60,
    "batch_freq" : 45,
    "epochs" : 75,
    "kfolds" : True,
    "seq_out" : False,
    "batch_size" : 128,
    "TBPTT" : False  # truncated back prop through time (not useful if the batches are shuffled anyways, or no longer in timeseries between batches) https://datascience.stackexchange.com/questions/118030/why-are-the-hidden-states-of-an-rnn-initialised-every-epoch-instead-of-every-bat
}

def test_loader(directory, multi_to_one = True):
    files_all = os.listdir(directory)

    Files_x = [file for file in files_all if "x.csv" in file]
    print(Files_x)
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
    prepend_empty = pd.DataFrame(np.zeros((_config_GRU["interval"]-1,7)))
    prepend_empty.columns = ["imu1","imu2","imu3","imu4","imu5","imu6","time"]
    for i, row in organizedDF.iterrows():
        X_data, Xt, y = [pd.read_csv(directory+file, header=None) for file in row]
        X = pd.concat([X_data, Xt], axis=1)
        X.columns = ["imu1","imu2","imu3","imu4","imu5","imu6","time"]
        if multi_to_one:
            X = pd.concat([prepend_empty, X], axis=0)
        X["pred"] = 0
        y.columns = ["time"]
        y['pred_y'] = 0
        y = y.reindex(columns=["pred_y", "time"])
        organizedX.append(X)
        organizedY.append(y)

    return organizedX, organizedY, Files_x

if __name__ == "__main__":
    testDir = "/home/xing/Classes/ECE542/Project/Projects-ECE542/Comp/data/TestData/"
    testDir = '/home/lixin/Classes/Spr23/542/Projects-ECE542/Comp/data/TestData/'
    model_path = "/home/lixin/Classes/Spr23/542/Projects-ECE542/Comp/Models/gru_best_1.pt"
    X_list, y_list, output_filenames = test_loader(testDir)

    for X_df in X_list:
        grumodel = model.GRUModel(6,
                                  interval=_config_GRU["interval"],
                                  hidden_size=_config_GRU["hidden"],
                                  num_layers=_config_GRU["num_stack_cells"],
                                  output_size=4, 
                                  all_output=_config_GRU["seq_out"]).cuda()
        grumodel.load_state_dict(torch.load(model_path))
        #print(X)
        X = X_df.to_numpy()
        X_preds = np.zeros((len(X),1))
        #print(len(X))
        #print(X_df.head())
        for i in range(0,len(X),_config_GRU["batch_size"]):
            if not _config_GRU["TBPTT"]:
                h0 = None
            batch = []
            for j in range(_config_GRU["batch_size"]):
                if (i+j+_config_GRU["interval"]) > len(X):
                    #print(i+j, len(X),i,j)
                    break
                data = np.array(X[i+j:i+j+_config_GRU["interval"]][:,:6]).reshape(_config_GRU["interval"],6)
                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0)
                data = np.nan_to_num((data - mean)/std, nan=0)

                #norm = np.linalg.norm(data, axis=1)
                #data = np.nan_to_num(data / norm,nan=0)
                batch.append(data) # single batch
            batch = np.array(batch)
            
            xinput = torch.Tensor(batch).view(-1, _config_GRU["interval"], 6).float().cuda()
            outputs, h1 = grumodel(xinput, h0)
            pred = torch.argmax(outputs, dim=1)
            X_preds[i+_config_GRU["interval"]-1:i+_config_GRU["batch_size"]+_config_GRU["interval"]-1] = pred.cpu().numpy()
            #print(i,(batch.shape), pred.cpu().item())
        #print(X_preds[:5])
        X_df["pred"] = X_preds
        #print(X_df.iloc[58:61])
        #print(X_df.iloc[-3:])
    fig, axes = plt.subplots(nrows=2, ncols=2)
    for i, y_df in enumerate(y_list):
        #print(y_df)
        merged = pd.merge(X_list[i][["time", "pred"]], y_df, how='outer', on='time', sort='True')
        #print(merged["pred"])
        if False:
            val_to_fill = np.where(pd.isnull(merged["pred"]))[0]
            ts = np.array([merged["pred"].loc[val_to_fill - i] for i in range(15)])
            #t_minus_4 = merged["pred"].loc[val_to_fill - 4]
            #t_minus_3 = merged["pred"].loc[val_to_fill - 3]
            #t_minus_2 = merged["pred"].loc[val_to_fill - 2]
            #t_minus_1 = merged["pred"].loc[val_to_fill - 1]
            #ts = np.array([t_minus_1, t_minus_2, t_minus_3, t_minus_4])
            #print(stats.mode(ts)[0])
            final = merged[merged["pred"].isna()].reset_index()
            #print(final)
            final["pred"] = pd.Series(stats.mode(ts)[0].flatten())
        else:
        #merged["pred"].fillna(pd.Series(stats.mode(ts)[0].flatten()))

            merged["pred"].interpolate(method='nearest', inplace=True)
            final = merged.dropna()[["time", "pred"]]
        output_filename = output_filenames[i].replace("x","y")
        final["pred"].to_csv(f"/home/lixin/Classes/Spr23/542/Projects-ECE542/Comp/data/TestData/generated/{output_filename}",
                            index= False,
                            header=False)
        #print(y_df)
        if i < 4:
            axes[int(i/2)][i%2].plot(final["time"], final["pred"], '.-b',linewidth = 1)
            axes[int(i/2)][i%2].set_title(output_filename)
        #y_df.plot(x="time", y="pred")
    plt.show()