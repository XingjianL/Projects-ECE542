# modification of
# https://www.kaggle.com/code/pmigdal/transfer-learning-with-resnet-50-in-pytorch

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
data_path = './TrainData/'
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
data_transforms = {
    'train':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]),
    'validation':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ]),
}
image_datasets = {
    'train': 
    datasets.ImageFolder(data_path + 'train', data_transforms['train']),
    'validation': 
    datasets.ImageFolder(data_path + 'validation', data_transforms['validation'])
}
dataloaders = {
    'train':
    torch.utils.data.DataLoader(image_datasets['train'],
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=0),  # for Kaggle
    'validation':
    torch.utils.data.DataLoader(image_datasets['validation'],
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=0)  # for Kaggle
}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(device)
for param in model.parameters():
    param.requires_grad = False   
    
model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 7)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters())


def train_model(model, criterion, optimizer, num_epochs=3):
    for epoch in range(num_epochs):
        batch_count = 0
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if batch_count % 100 == 0: print(f"batch:{batch_count}/{np.ceil(len(image_datasets[phase])/batch_size)}, batch loss: {loss.item()}")
                    batch_count += 1

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                        epoch_loss,
                                                        epoch_acc))
    return model

start = time.time()
model_trained = train_model(model, criterion, optimizer, num_epochs=3)
end = time.time()
print(f"Time taken: {end-start} seconds")
