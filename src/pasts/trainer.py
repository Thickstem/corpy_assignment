import os
import sys 
import glob 

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch 
import torch.nn as nn 
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optims
from torchvision import transforms

from PIL import Image
from sklearn.metrics import confusion_matrix,accuracy_score
import matplotlib.pyplot as plt

class Dataset(data.Dataset):

    def __init__(self,data,transform=None):
        self.dataset=data # [path,label]のpdDF
        self.transform = transform
        
    def __len__(self):
        return self.dataset.shape[0]
    
    def __getitem__(self,idx):
        
        img = Image.open(self.dataset.iloc[idx]["path"])
        img = self.transform(img)

        return img,self.dataset.iloc[idx]["label"]

class CNN(nn.Module):
    def __init__(self,num_classes):
        super(CNN,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Linear(in_features= 32 * 32 * 64, out_features=num_classes)
    
    def forward(self, x):
        x = self.features(x)
        #print(x.size())
        x = x.view(x.size(0), -1)
        #print(x.size())
        x = self.classifier(x)
        return x

if __name__=="__main__":
    path_label = pd.read_csv("binary_label.csv",index_col=0)
    transform = transforms.Compose(
        [transforms.Resize(256),
        transforms.ToTensor()]
    )
    train_dataset = Dataset(path_label,transform)
    dataloader = train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=8,shuffle=True,num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN(num_classes=2)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

    n_epoch= 10 
    for epoch in range(n_epoch):
        tmp_loss = 0.0

        for input,labels in tqdm(train_loader):
            input = input.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(input)
            
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            tmp_loss+=loss.item()
        
        tmp_loss/=len(train_dataset)
        print(f"epoch:{epoch}, Loss:{loss}")
    
    with torch.no_grad():
        
