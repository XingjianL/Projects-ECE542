import copy
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

print(torch.__version__) #1.12.0+cu116
import torchvision
print(torchvision.__version__) #0.13.0+cu116

batch_size = 32

test_data_path = '/home/lixin/Classes/Spr23/542/Projects-ECE542/Group/TestData/'
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

data_transforms = {
    'test':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ])
}
image_datasets = {
    'test': 
    datasets.ImageFolder(test_data_path + 'test', data_transforms['test'])
}
dataloaders = {
    'test':
    torch.utils.data.DataLoader(image_datasets['test'],
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=0)  # for Kaggle
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_path = "/home/lixin/Classes/Spr23/542/Projects-ECE542/Group/Models/last.pt"
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(device)
for param in model.parameters():
    param.requires_grad = False   
    
model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 7)).to(device)
model.load_state_dict(torch.load(model_path))

def test_model(model):
    model.eval()
    corrects = 0
    incorrects = 0
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        corrects += torch.sum(preds == labels.data)
        incorrects += torch.sum(preds != labels.data)
        with open('Group/resnet50transfer_preds.csv', 'a') as fd:
            for i in range(preds.size(0)):
                numpy_arr = outputs[i].cpu().detach().numpy()
                output_str = ",".join(["%.5f" % num for num in numpy_arr])
                output_str = "\n" + output_str + f",{preds[i]},{labels[i]}" 
                fd.write(output_str)
    return corrects, incorrects

def confusion_matrix(preds_csv_filename = 'Group/resnet50transfer_preds.csv'):
    df = pd.read_csv(preds_csv_filename, header=None)
    preds = df.iloc[:,-2].to_numpy()
    labels = df.iloc[:,-1].to_numpy()
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(labels,preds)

correct, incorrect = test_model(model)
print(confusion_matrix())