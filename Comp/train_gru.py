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
from sklearn.model_selection import KFold
_config_GRU = {
    "val_split" : 5/10,
    "lr" : 0.001,
    "min_lr" : 0.0001,
    "hidden" : 128,
    "num_stack_cells" : 2,
    "interval" : 60,
    "batch_freq" : 55,
    "epochs" : 40,
    "kfolds" : True,
    "seq_out" : False,
    "batch_size" : 128,
    "TBPTT" : False  # truncated back prop through time (not useful if the batches are shuffled anyways, or no longer in timeseries between batches) https://datascience.stackexchange.com/questions/118030/why-are-the-hidden-states-of-an-rnn-initialised-every-epoch-instead-of-every-bat
}
class SimpleDataset(data.Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
def generateTrainValDataloader(train_dir):
    if _config_GRU["kfolds"] is True:
        train_set, _ = vis.prepareData(train_dir, 
                                             plot=False,
                                             val_split=0)
        train_set_x, train_set_y = vis.prepareBatchedData(train_set,
                                                          interval = _config_GRU["interval"],
                                                          batch_freq = _config_GRU["batch_freq"],
                                                          randomize = _config_GRU["batch_freq"] != 1,
                                                          noise = False,
                                                          seq_output = _config_GRU["seq_out"],
                                                          force_all=False)
        train_set_x = torch.Tensor(train_set_x)
        train_set_y = torch.Tensor(train_set_y)
        _, train_class_weights = torch.unique(torch.argmax(train_set_y,dim=2),return_counts = True)
        train_class_weights = max(train_class_weights)/train_class_weights
        return SimpleDataset(train_set_x, train_set_y), _, train_class_weights
    
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
    train_set_x = torch.Tensor(train_set_x)
    train_set_y = torch.Tensor(train_set_y)

    # weights for cross entropy loss
    _, train_class_weights = torch.unique(torch.argmax(train_set_y,dim=2),return_counts = True)
    train_class_weights = max(train_class_weights)/train_class_weights
    #train_class_weights[3] = train_class_weights[3] * 2
    print(train_class_weights)
    train_dataloader = data.DataLoader(
        data.TensorDataset(train_set_x, train_set_y),
        batch_size=_config_GRU["batch_size"],
        shuffle=True,
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

def nonKFoldTrain(gru_model, criterion, optimizer, scheduler, train_dataloader, val_dataloader):
    epoch_train_loss = []
    epoch_val_loss = []
    epoch_val_f1 = []
    max_val_f1 = 0
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
        scheduler.step()
        print(f"epoch {epoch}, end, average batch loss {train_loss/i}")

        # SAVE MODEL
        torch.save(gru_model.state_dict(), "/home/lixin/Classes/Spr23/542/Projects-ECE542/Comp/Models/gru_last.pt")
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
        print(f"max f1 {max_val_f1}, lr: {scheduler.get_last_lr()}")
        epoch_val_f1.append(np.mean(fscore))
        if np.mean(fscore) > max_val_f1:
            max_val_f1 = np.mean(fscore)
            torch.save(copy.deepcopy(gru_model.state_dict()), "/home/lixin/Classes/Spr23/542/Projects-ECE542/Comp/Models/gru_best.pt")
    plt.plot(epoch_train_loss)
    plt.plot(epoch_val_loss)
    plt.plot(epoch_val_f1)
    plt.show()

def KFoldTrain(dataloader, weights):
    num_folds = int(1 / _config_GRU["val_split"])
    kfold = KFold(n_splits=num_folds, shuffle=True)
    epoch_train_loss = []
    epoch_val_loss = []
    epoch_val_f1 = []
    max_val_f1 = np.zeros(num_folds)
    for fold, (train_id, val_id) in enumerate(kfold.split(dataloader)):
        gru_model = model.GRUModel(num_inputs,
                               interval=_config_GRU["interval"],
                               hidden_size=_config_GRU["hidden"],
                               num_layers=_config_GRU["num_stack_cells"],
                               output_size=4,
                               all_output=_config_GRU["seq_out"]).cuda()
        optimizer = torch.optim.AdamW(gru_model.parameters(), lr=_config_GRU["lr"], weight_decay=1e-6)
        T_max = int(_config_GRU["epochs"]*_config_GRU["val_split"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=_config_GRU["min_lr"])
        criterion = nn.CrossEntropyLoss(weight = weights).cuda()

        print('-'*20)
        print(f"Fold: {fold}")
        print(f"train, val batch counts: {len(train_id)}, {len(val_id)}")
        train_sampler = data.SubsetRandomSampler(train_id)
        val_sampler = data.SubsetRandomSampler(val_id)
        train_loader = data.DataLoader(dataloader, batch_size=_config_GRU["batch_size"], sampler=train_sampler)
        valid_loader = data.DataLoader(dataloader, batch_size=_config_GRU["batch_size"], sampler=val_sampler)

        for epoch in range(int(_config_GRU["epochs"]/num_folds)):
            print()
            train_loss = 0
            last_loss = 0

            h0 = None
            # each training epoch
            for i, (seqs, labels) in enumerate(train_loader):
                #print(labels.shape)
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
            scheduler.step()
            print(f"epoch {epoch}, end, average batch loss {train_loss/i}")

            # SAVE MODEL
            torch.save(gru_model.state_dict(), "/home/lixin/Classes/Spr23/542/Projects-ECE542/Comp/Models/gru_last.pt")
            epoch_train_loss.append(train_loss/i)
            # each valid epoch
            valid_loss = 0
            running_label = []
            running_pred = []
            h0 = None
            for i, (seqs, labels) in enumerate(valid_loader):
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
            print(f"max f1 {max_val_f1[fold]}, lr: {scheduler.get_last_lr()}")
            epoch_val_f1.append(np.mean(fscore))
            if np.mean(fscore) > max_val_f1[fold]:
                max_val_f1[fold] = np.mean(fscore)
                torch.save(copy.deepcopy(gru_model.state_dict()), f"/home/lixin/Classes/Spr23/542/Projects-ECE542/Comp/Models/gru_best_{fold}.pt")
    print("F1 for each fold: ", max_val_f1, " average: ", np.mean(max_val_f1))
    plt.plot(epoch_train_loss)
    plt.plot(epoch_val_loss)
    plt.plot(epoch_val_f1)
    plt.show()
    return

if __name__ == "__main__":
    random.seed(1)
    torch.manual_seed(1)
    torch.autograd.set_detect_anomaly(True)
    train_dir = '/home/xing/Classes/ECE542/Project/Projects-ECE542/Comp/data/TrainingData/'
    train_dir = '/home/lixin/Classes/Spr23/542/Projects-ECE542/Comp/data/TrainingData/'
    
    train_dataloader, val_dataloader, weights = generateTrainValDataloader(train_dir)

    num_inputs = 6

    gru_model = model.GRUModel(num_inputs,
                               interval=_config_GRU["interval"],
                               hidden_size=_config_GRU["hidden"],
                               num_layers=_config_GRU["num_stack_cells"],
                               output_size=4,
                               all_output=_config_GRU["seq_out"]).cuda()
    criterion = nn.CrossEntropyLoss(weight = weights).cuda()
    optimizer = torch.optim.AdamW(gru_model.parameters(), lr=_config_GRU["lr"], weight_decay=1e-6)
    T_max = int(_config_GRU["epochs"]/4)
    if _config_GRU["kfolds"]:
        T_max = int(_config_GRU["epochs"]*_config_GRU["val_split"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=_config_GRU["min_lr"])

    if _config_GRU["kfolds"] is False:
        nonKFoldTrain(gru_model, criterion, optimizer, scheduler, train_dataloader, val_dataloader)
    else:
        KFoldTrain(train_dataloader, weights)