import numpy as np
import torch 
import torch.utils.data as data
from torchvision import transforms
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optims

from PIL import Image

class Dataset(data.Dataset):

    def __init__(self,data,transform=None):
        self.dataset=data # [path,label]のpdDF
        self.transform = transform
        
    def __len__(self):
        return self.dataset.shape[0]
    
    def __getitem__(self,idx):
        
        img = Image.open(self.dataset.iloc[idx]["path"])
        img = img.convert("RGB")
        #img = cv2.imread(self.dataset.iloc[idx]["path"])
        img = self.transform(img)

        return img,self.dataset.iloc[idx]["label"]
