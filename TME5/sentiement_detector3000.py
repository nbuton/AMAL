import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tp5_preprocess import *
import gzip
from torch.utils.data import DataLoader

class Sentiment(nn.Module):
    def __init__(self, taille_embedding, nb_feature_map ,taille_dico):
        super(Sentiment, self).__init__()
        self.embed = nn.Embedding(taille_dico,taille_embedding)
        self.conv = nn.Conv1d(taille_embedding,nb_feature_map,3,1)
        self.maxPool = nn.MaxPool1d(3,2)

    def forward(self, x):
        #print("x debut",x.shape)
        x = self.embed(x)
        #print("after embed",x.shape)
        x = x.transpose(1,2)
        #print("apres transpose",x.shape)
        x = self.conv(x)
        x = self.maxPool(x)
        #print(x.shape)
        x = torch.max(x,axis=1)[1]
        #print(x.shape)
        return x

    
def main_sentiment(train,test):
    taille_embedding = 50
    nb_feature_map = 5
    taille_dico = 1000
    myCNN = Sentiment( taille_embedding, nb_feature_map ,taille_dico)
    loss = nn.MSELoss()
    optim = Adam(myCNN.parameters(), lr=1e-4)
    for data, target in train:
        #print(type(data))
        #print(data.shape)
        out = myCNN(data)
        #print(out.shape)

def loaddata(f):
    with gzip.open(f,"rb") as fp:
        return torch.load(fp)

def main():
    b_size = 32
    train = DataLoader(loaddata("train-1000.pth"),batch_size=b_size,collate_fn=TextDataset.collate)
    test = DataLoader(loaddata("test-1000.pth"),batch_size=b_size,collate_fn=TextDataset.collate)
    main_sentiment(train,test)

if __name__ == '__main__':
    main()