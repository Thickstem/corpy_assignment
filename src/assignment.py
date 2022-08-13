import random
from random import sample
import logging
import argparse
from turtle import back
import numpy as np
import pandas as pd

from tqdm import tqdm
from PIL import Image
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import wide_resnet50_2, resnet18



random.seed(1024)
torch.manual_seed(1024)
torch.cuda.manual_seed_all(1024)

def setup_logger(name, logfile):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # create file handler which logs even DEBUG messages
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    #fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(name)s - %(funcName)s - %(message)s')
    #fh.setFormatter(fh_formatter)

    # create console handler with a INFO log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
    ch.setFormatter(ch_formatter)

    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


class Dataset(data.Dataset):

    def __init__(self,data,transform=None):
        self.dataset=data # [path,label]のpdDF
        self.transform = transform
        
    def __len__(self):
        return self.dataset.shape[0]
    
    def __getitem__(self,idx):
        
        img = Image.open(self.dataset.iloc[idx]["path"]).convert("RGB")
        img = self.transform(img)
        
        return img,self.dataset.iloc[idx]["label"]

def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z

def train(dataloader,model,sampling_idx):
    outputs = []
    def hook(module, input, output):
            outputs.append(output)
    
    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)

    train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
    for input,label in tqdm(dataloader): 
        with torch.no_grad():
            _ = model(input.to(device))
        
        for k,v in zip(train_outputs.keys(),outputs):
            train_outputs[k].append(v.cpu().detach())
        
        outputs=[]

    for k,v in train_outputs.items():
        train_outputs[k]=torch.cat(v,0)

    # Embedding concat
    embedding_vectors = train_outputs["layer1"]
    for layer_name in ["layer2", "layer3"]:
        embedding_vectors = embedding_concat(embedding_vectors,train_outputs[layer_name])
    
    # randomly select d dimension
    embedding_vectors = torch.index_select(embedding_vectors, 1, sampling_idx)
    # calculate multivariate Gaussian distribution
    B,C,H,W = embedding_vectors.size()
    embedding_vectors = embedding_vectors.view(B,C,H*W)
    mean=torch.mean(embedding_vectors,dim=0).numpy()
    cov = torch.zeros(C,C,H*W).numpy()
    I  = np.identity(C)
    for i in range(H*W):
        cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I
    # save learned distribution
    train_outputs = [mean, cov]
    
    return train_outputs
    
    
def test(dataloader,model,sampling_idx,train_outputs):

    outputs = []
    def hook(module, input, output):
            outputs.append(output)
    
    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)
    

    test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

    ans_list = []
    test_imgs = []
    
    for input,label in tqdm(dataloader):
        test_imgs.extend(input.cpu().detach().numpy())
        ans_list.extend(label.cpu().detach().numpy())
        with torch.no_grad():
            _ = model(input.to(device))
        
        for k,v in zip(test_outputs.keys(),outputs):
            test_outputs[k].append(v.cpu().detach())
        
        outputs =[]
    
    for k,v in test_outputs.items():
        test_outputs[k] = torch.cat(v,0)

    embedding_vectors = test_outputs["layer1"]
    for layer_name in ["layer2","layer3"]:
        embedding_vectors = embedding_concat(embedding_vectors,test_outputs[layer_name])

    # randomly select d dimension
    embedding_vectors = torch.index_select(embedding_vectors,1,sampling_idx)

    B,C,H,W = embedding_vectors.size()
    embedding_vectors = embedding_vectors.view(B,C,H*W).numpy()
    dist_list=[]

    for i in range(H*W):
        mean = train_outputs[0][:,i]
        conv_inv = np.linalg.inv(train_outputs[1][:,:,i])
        dist = [mahalanobis(sample[:,i],mean,conv_inv) for sample in embedding_vectors]
        dist_list.append(dist)
    dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

    # upsample
    dist_list = torch.tensor(dist_list)
    score_map = F.interpolate(dist_list.unsqueeze(1),size=input.size(2),mode="bilinear",
                                align_corners=False).squeeze().numpy()

    for i in range(score_map.shape[0]):
        score_map[i] = gaussian_filter(score_map[i],sigma=4)

    max_score = score_map.max()
    min_score = score_map.min()
    scores = (score_map - min_score) / (max_score - min_score)

    # calculate image-level ROC AUC score
    img_scores = scores.reshape(scores.shape[0],-1).max(axis=1)
    ans_list = np.asarray(ans_list)
    fpr,tpr,_ = roc_curve(ans_list,img_scores)
    roc_auc = roc_auc_score(ans_list,img_scores)
    
    return fpr,tpr,roc_auc


if __name__ =="__main__":
    logger = setup_logger(name="screw")

    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone",type=str,default="wide_resnet50_2")
    args = parser.parse_args()

    logger = setup_logger(name="screw",logfile=f"screw_{args.backbone}")
    
    labels = pd.read_csv("train_label.csv",index_col=0)
    val_data = labels.iloc[:100]
    train_data = labels.iloc[100:]

    transform = transforms.Compose([
    transforms.Resize(256,Image.ANTIALIAS),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
    ])


    train_dataset = Dataset(train_data,transform)
    val_dataset = Dataset(val_data,transform)

    train_loader = DataLoader(train_dataset,batch_size=8,shuffle=True,num_workers=2)
    val_loader = DataLoader(val_dataset,batch_size=8,shuffle=False,num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )

    backbone = args.backbone
    logger.debug(f"backbone:{backbone}")
    
    if backbone == 'resnet18':
        model = resnet18(pretrained=True, progress=True)
        t_d = 448
        d = 100
    elif backbone == 'wide_resnet50_2':
        model = wide_resnet50_2(pretrained=True, progress=True)
        t_d = 1792
        d = 550
    model.to(device)
    model.eval()

    sampling_idx = torch.tensor(sample(range(0, t_d), d)) # for random dimension sampling 
    
    logger.debug("Generating train features")
    train_outputs = train(train_loader,model,sampling_idx)
    logger.debug("Now Evaluating ...")
    fpr,tpr,roc_auc = test(val_loader,model,sampling_idx,train_outputs)

    logger.debug(f"ROCAUC score:{roc_auc:.4f}")
    plt.plot(fpr,tpr,label="Screw")
    plt.title(f"ROCAUC ({backbone})")
    plt.legend()
    plt.grid()
    plt.savefig("screw_ROCAUC")

    




    
    
    


    

