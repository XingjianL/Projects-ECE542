import random

import vis

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ncps.wirings import AutoNCP
from ncps.torch import LTC, CfC
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
import torch
import torch.utils.data as data

import matplotlib.pyplot as plt
import seaborn as sns

_config = {
    "val_split" : 1/10,
    "lr" : 0.002,
    "hidden" : 128,
    "activation" : "silu",
    "back_dr" : 0.5,
    "back_layer" : 2,
    "back_units" : 64,
    "interval" : 80,
    "batch_freq" : 1,
    "epochs" : 50
}

# LightningModule for training a RNNSequence module
class SequenceLearner(pl.LightningModule):
    def __init__(self, model, lr=0.005, class_weights = None):
        super().__init__()
        self.model = model
        self.lr = lr
        self.class_weights = class_weights.cuda() # imbalanced data for weighted cross entropy
        print(class_weights)
        self.accs = []
        self.highest_acc = 0
        #self.class_weights = None
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.model.forward(x)
        y_hat = y_hat.view_as(y)
        pred = torch.argmax(y_hat, dim=2)

        #loss = nn.MSELoss()(y_hat, y)
        y_hat = y_hat.swapaxes(1,2)
        target = torch.argmax(y, dim=2)
        loss = nn.CrossEntropyLoss(weight=self.class_weights)(y_hat, target)
        acc = accuracy(pred.view(-1), target.view(-1), 'multiclass', num_classes=4)
        self.log("train_acc", acc, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.model.forward(x)
        y_hat = y_hat.view_as(y)
        pred = torch.argmax(y_hat, dim=2)

        #loss = nn.MSELoss()(y_hat, y)
        y_hat = y_hat.swapaxes(1,2)
        target = torch.argmax(y, dim=2)
        loss = nn.CrossEntropyLoss(weight=self.class_weights)(y_hat, target)
        acc = accuracy(pred.view(-1), target.view(-1), 'multiclass', num_classes=4)
        self.accs.append(acc.cpu())
        self.log("val_loss", loss, prog_bar=True)
        return loss
    def on_validation_epoch_end(self):
        acc = np.mean(self.accs)
        self.accs = []
        self.log("val_acc", acc, prog_bar=True)
        if acc > self.highest_acc:
            self.highest_acc = acc
            self.savebestmodel()
        self.savelastmodel()
        return
    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=4e-06)

    def savebestmodel(self):
        torch.save(self.model.state_dict(), f"Comp/Models/{self.current_epoch}_val_{self.highest_acc}.pt")
        return
    def savelastmodel(self):
        torch.save(self.model.state_dict(), f"Comp/Models/last.pt")
        return
if __name__ == "__main__":
    random.seed(1)
    train_dir = '/home/xing/Classes/ECE542/Project/Projects-ECE542/Comp/data/TrainingData/'
    train_set, val_set = vis.prepareData(train_dir, 
                                         plot=False,
                                         val_split=_config["val_split"])
    print(train_set[0])
    print(len(train_set[0]))
    train_set_x, train_set_y = vis.prepareBatchedData(train_set,
                                                      interval = _config["interval"],
                                                      batch_freq = _config["batch_freq"],
                                                      randomize = True,
                                                      noise = True)
    train_set_x = torch.Tensor(train_set_x)
    train_set_y = torch.Tensor(train_set_y)

    # weights for cross entropy loss
    _, train_class_weights = torch.unique(torch.argmax(train_set_y,dim=2),return_counts = True)
    train_class_weights = max(train_class_weights)/train_class_weights

    train_dataloader = data.DataLoader(
        data.TensorDataset(train_set_x, train_set_y),
        batch_size=128,
        shuffle=True,
        num_workers=4
    )

    val_set_x, val_set_y = vis.prepareBatchedData(val_set, 
                                                  interval=_config["interval"], 
                                                  batch_freq=1,
                                                  randomize = False,
                                                  noise=False)
    val_set_x = torch.Tensor(val_set_x)
    val_set_y = torch.Tensor(val_set_y)
    val_dataloader = data.DataLoader(
        data.TensorDataset(val_set_x, val_set_y),
        batch_size=128,
        shuffle=False,
        num_workers=4
    )

    sns.set()
    fig, axes = plt.subplots(3,3)
    for i in range(3):
        for j in range(3):
            axes[i,j].plot(train_set_x[100*(i*3+j), :, :],label="IMU")
            #plt.plot(train_set_x[0, :, 1], label="Input feature 1")
            axes[i,j].plot(train_set_y[100*(i*3+j), :, :], label="Label")
            #axes[i,j].ylim((-1.1, 1.1))
            #axes[i,j].title("Training data")
            #axes[i,j].legend(loc="upper right")
    plt.show()

    #wiring = AutoNCP(32, 4, sparsity_level=0.5)  # 32 units, 4 motor neuron

    # 0.80 max val acc @ epoch 83 and 20% val, no data augmentation besides Y interpolation
    #cfc_model = CfC(6, 128, proj_size=4, batch_first=True, activation="silu", backbone_dropout=0.2, backbone_layers=2, backbone_units=64)
    
    # 0.85 max val acc @ epoch 24 and 10% val, no aug
    #cfc_model = CfC(6, 64, proj_size=4, batch_first=True, activation="relu", backbone_dropout=0.2, backbone_layers=2, backbone_units=128)
    
    cfc_model = CfC(6, 
                    _config["hidden"],
                    proj_size=4, 
                    batch_first=True, 
                    activation=_config["activation"], 
                    backbone_dropout=_config["back_dr"], 
                    backbone_layers=_config["back_layer"], 
                    backbone_units=_config["back_units"])

    learn = SequenceLearner(cfc_model, 
                            lr=_config["lr"], 
                            class_weights=train_class_weights)
    
    trainer = pl.Trainer(
        logger=pl.loggers.CSVLogger("log"),
        max_epochs=_config["epochs"],
        gradient_clip_val=1,  # Clip gradient to stabilize training

        accelerator="gpu", # use gpu
        devices="auto"
    )
    #sns.set_style("white")
    #plt.figure(figsize=(6, 4))
    #legend_handles = wiring.draw_graph(draw_labels=True, neuron_colors={"command": "tab:cyan"})
    #plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
    #sns.despine(left=True, bottom=True)
    #plt.tight_layout()
    #plt.show()
    # Let's visualize how LTC initialy performs before the training
    sns.set()
    with torch.no_grad():
        prediction = cfc_model(train_set_x)[0].numpy()
    plt.figure(figsize=(6, 4))
    plt.plot(train_set_y[100, :, :], label="Target output")
    plt.plot(prediction[100, :, :], label="NCP output")
    plt.ylim((-1.1, 1.1))
    plt.title("Before training")
    plt.legend(loc="upper right")
    plt.show()

    # Train the model for 400 epochs (= training steps)
    trainer.fit(learn, train_dataloader,val_dataloader)
    # How does the trained model now fit to the sinusoidal function?
    fig, axes = plt.subplots(3,3)
    with torch.no_grad():
        prediction = cfc_model(train_set_x)[0].numpy()
    for i in range(3):
        for j in range(3):
            axes[i,j].plot(train_set_y[100*i, :, :], label="Target output")
            #plt.plot(train_set_x[0, :, 1], label="Input feature 1")
            axes[i,j].plot(prediction[100*i, :, :], label="NCP output")
            #axes[i,j].ylim((-1.1, 1.1))
            #axes[i,j].title("Training data")
            #axes[i,j].legend(loc="upper right")
    plt.show()