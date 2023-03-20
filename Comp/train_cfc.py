import random

import vis

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ncps.wirings import AutoNCP
from ncps.torch import LTC, CfC
import pytorch_lightning as pl
import torch
import torch.utils.data as data

import matplotlib.pyplot as plt
import seaborn as sns

# LightningModule for training a RNNSequence module
class SequenceLearner(pl.LightningModule):
    def __init__(self, model, lr=0.005):
        super().__init__()
        self.model = model
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.model.forward(x)
        y_hat = y_hat.view_as(y)
        loss = nn.MSELoss()(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.model.forward(x)
        y_hat = y_hat.view_as(y)
        loss = nn.MSELoss()(y_hat, y)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

if __name__ == "__main__":
    random.seed(1)
    train_dir = '/home/lixin/Classes/Spr23/542/Projects-ECE542/Comp/data/TrainingData/'
    train_set, val_set = vis.prepareData(train_dir, plot=False)
    print(train_set[0])
    print(len(train_set[0]))
    #N = len(train_set[0])
    N = 40 # 1 second interval
    batch_count = np.floor(len(train_set[0])/N).astype(int)
    print(batch_count)
    train_set_x = train_set[0].to_numpy()[:batch_count*N,:6].reshape([batch_count,N,6]).astype(np.float32)
    train_set_y = train_set[0].to_numpy()[:batch_count*N,8:].reshape([batch_count,N,4]).astype(np.float32)
    #print(train_set[0].to_numpy()[start:start+N:,-1])
    print("data_x.shape: ", str(train_set_x.shape))
    print("data_y.shape: ", str(train_set_y.shape))
    train_set_x = F.normalize(torch.Tensor(train_set_x))
    train_set_y = torch.Tensor(train_set_y)

    dataloader = data.DataLoader(
        data.TensorDataset(train_set_x, train_set_y),
        batch_size=128,
        #shuffle=True,
        num_workers=4
    )
    for i in range(10):
        sns.set()
        plt.figure(figsize=(6, 4))
        plt.plot(train_set_x[100*i, :, :],label="IMU")
        #plt.plot(train_set_x[0, :, 1], label="Input feature 1")
        plt.plot(train_set_y[100*i, :, :], label="Label")
        plt.ylim((-1.1, 1.1))
        plt.title("Training data")
        plt.legend(loc="upper right")
        plt.show()

    wiring = AutoNCP(16, 4)  # 16 units, 4 motor neuron

    cfc_model = CfC(6, wiring, batch_first=True)
    learn = SequenceLearner(cfc_model, lr=0.01)
    trainer = pl.Trainer(
        logger=pl.loggers.CSVLogger("log"),
        max_epochs=200,
        gradient_clip_val=1,  # Clip gradient to stabilize training
        accelerator="gpu", 
        devices="auto"
    )

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
    trainer.fit(learn, dataloader)

    # How does the trained model now fit to the sinusoidal function?
    for i in range(10):
        sns.set()
        with torch.no_grad():
            prediction = cfc_model(train_set_x)[0].numpy()
        plt.figure(figsize=(6, 4))
        plt.plot(train_set_y[100*i, :, :], label="Target output")
        plt.plot(prediction[100*i, :, :], label="NCP output")
        plt.ylim((-1.1, 1.1))
        plt.title("After training")
        plt.legend(loc="upper right")
        plt.show()