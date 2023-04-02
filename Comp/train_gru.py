import copy
import vis
import torch
import torch.nn as nn
import model
import random
import torch.utils.data as data
from torch.autograd import Variable
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
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

def generateTrainValDataloader(train_dir):
    train_set, val_set = vis.prepareData(train_dir, 
                                         plot=False,
                                         val_split=_config_GRU["val_split"])
    print(len(train_set))
    print(len(train_set[0]))
    train_set_x, train_set_y = vis.prepareBatchedData(train_set,
                                                      interval = _config_GRU["interval"],
                                                      batch_freq = _config_GRU["batch_freq"],
                                                      randomize = _config_GRU["batch_freq"] != 1,
                                                      noise = True,
                                                      seq_output = _config_GRU["seq_out"],
                                                      force_all=False)
    if _config_GRU["freq"]:
        train_set_x = vis.frequencyDomain(train_set_x, only_fft=_config_GRU["freq_only"])
    train_set_x = torch.Tensor(train_set_x)
    train_set_y = torch.Tensor(train_set_y)

    # weights for cross entropy loss
    _, train_class_weights = torch.unique(torch.argmax(train_set_y,dim=2),return_counts = True)
    train_class_weights = max(train_class_weights)/train_class_weights
    train_class_weights[3:] = train_class_weights[3:] * 1.5
    print(train_class_weights)
    train_dataloader = data.DataLoader(
        data.TensorDataset(train_set_x, train_set_y),
        batch_size=_config_GRU["batch_size"],
        shuffle=False,
        num_workers=4,
        #drop_last=True
    )

    val_set_x, val_set_y = vis.prepareBatchedData(val_set, 
                                                  interval=_config_GRU["interval"], 
                                                  batch_freq=1,
                                                  randomize = False,
                                                  noise=False,
                                                  seq_output=_config_GRU["seq_out"],
                                                  force_all=True)
    if _config_GRU["freq"]:
        val_set_x = vis.frequencyDomain(val_set_x, only_fft=_config_GRU["freq_only"])
    val_set_x = torch.Tensor(val_set_x)
    val_set_y = torch.Tensor(val_set_y)
    val_dataloader = data.DataLoader(
        data.TensorDataset(val_set_x, val_set_y),
        batch_size=_config_GRU["batch_size"],
        shuffle=False,
        num_workers=4,
        #drop_last=True
    )
    return train_dataloader, val_dataloader, train_class_weights

if __name__ == "__main__":
    random.seed(1)
    torch.manual_seed(1)
    torch.autograd.set_detect_anomaly(True)
    train_dir = '/home/xing/Classes/ECE542/Project/Projects-ECE542/Comp/data/TrainingData/'
    train_dir = '/home/lixin/Classes/Spr23/542/Projects-ECE542/Comp/data/TrainingData/'
    
    train_dataloader, val_dataloader, weights = generateTrainValDataloader(train_dir)

    if not _config_GRU["freq_only"] and _config_GRU["freq"]:
        num_inputs = 12
    else:
        num_inputs = 6

    gru_model = model.GRUModel(num_inputs,hidden_size=_config_GRU["hidden"],num_layers=_config_GRU["num_stack_cells"],output_size=4, all_output=_config_GRU["seq_out"]).cuda()
    criterion = nn.CrossEntropyLoss(weight = weights).cuda()
    optimizer = torch.optim.AdamW(gru_model.parameters(), lr=_config_GRU["lr"], weight_decay=1e-6)
    

    epoch_train_loss = []
    epoch_val_loss = []
    for epoch in range(_config_GRU["epochs"]):
        print()
        train_loss = 0
        last_loss = 0
        
        h0 = None
        # each training epoch
        for i, (seqs, labels) in enumerate(train_dataloader):
            if not _config_GRU["TBPTT"]:
                h0 = None
            seqs = Variable(seqs.view(-1, _config_GRU["interval"], num_inputs).cuda())
            if _config_GRU["seq_out"] is True:
                labels = Variable(torch.argmax(labels.view(-1, _config_GRU["interval"], 4),dim=2).cuda())
            else:
                labels = Variable(torch.argmax(labels.view(-1, 1, 4),dim=2).cuda())
            optimizer.zero_grad()
            outputs, h1 = gru_model(seqs, h0)
            h0 = h1.detach()
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            train_loss += loss.item()
            if i % 1000 == 999:
                last_loss = train_loss / i
                print(f"epoch {epoch}, batch {i}, avgloss {last_loss}")
        print(f"epoch {epoch}, end, average batch loss {train_loss/i}")
        epoch_train_loss.append(train_loss/i)
        # each valid epoch
        valid_loss = 0
        running_label = []
        running_pred = []
        h0 = None
        for i, (seqs, labels) in enumerate(val_dataloader):
            if not _config_GRU["TBPTT"]:
                h0 = None
            seqs = Variable(seqs.view(-1, _config_GRU["interval"], num_inputs).cuda())
            if _config_GRU["seq_out"] is True:
                labels = Variable(torch.argmax(labels.view(-1, _config_GRU["interval"], 4),dim=2).cuda())
            else:
                labels = Variable(torch.argmax(labels.view(-1, 1, 4),dim=2).cuda())
            outputs, h1 = gru_model(seqs, h0)
            h0 = h1.detach()
            loss = criterion(outputs, labels)

            valid_loss += loss.item()

            #print(outputs.shape)
            #print(labels.shape)
            pred = torch.argmax(outputs, dim=1)
            #target = torch.argmax(labels, dim=2)

            running_label.append(copy.deepcopy(labels.view(-1).cpu().numpy()))
            running_pred.append(copy.deepcopy(pred.view(-1).cpu().numpy()))

        precision, recall, fscore, support = metrics.precision_recall_fscore_support(np.concatenate(running_label), np.concatenate(running_pred))
        print(f"validation epoch {epoch}, average valid loss {valid_loss/i}")
        print(f"validation f1: {fscore}, average: {np.mean(fscore)}")
        epoch_val_loss.append(valid_loss/i)

    plt.plot(epoch_train_loss)
    plt.plot(epoch_val_loss)
    plt.show()