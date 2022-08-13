import torch
import torch.nn as nn
import torch.functional as F

class CNN(nn.Module):
    def __init__(self,num_classes):
        super(CNN,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc1 = nn.Linear(in_features= 28 * 28 * 64, out_features=128)
        self.classifier = nn.Linear(in_features=128,out_features=num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        #print(x.size())
        x = self.fc1(x)
        x = self.classifier(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self,z_dim):
        super(Autoencoder,self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(224*224,2048),
            nn.ReLU(True),
            nn.Linear(2048,512),
            nn.ReLU(True),
            nn.Linear(512,128),
            nn.ReLU(True),
            nn.Linear(128,z_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim,128),
            nn.ReLU(True),
            nn.Linear(128,512),
            nn.ReLU(True),
            nn.Linear(512,2048),
            nn.ReLU(True),
            nn.Linear(2048,224*224),
            nn.Tanh()  
        )
    
    def forward(self,x):
        z = self.encoder(x)
        xhat = self.decoder(z)
        return xhat