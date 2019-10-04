import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 

class MonDataset(Dataset):
    def __init__(self,nameCsv,nameLabel):
        data = pd.read_csv(nameCsv)
        data = data.sample(frac=1).reset_index(drop=True)
        #self.w2 = theano.shared((np.random.randn(10, 1, 3, 3)).astype(theano.config.floatX))
        self.all_data_target = torch.Tensor(data[nameLabel].values.astype(np.float64))
        self.all_data = torch.Tensor(data.drop(nameLabel, axis = 1).values.astype(np.float64))

    def __getitem__(self,index):
        return self.all_data[index],self.all_data_target[index]
    
    def __len__(self):
        return len(self.all_data)

class AutoEncoder(nn.Module):
    def __init__(self,din,dout):
        super(AutoEncoder,self).__init__()
        self.enc = nn.Linear(din,dout)
        self.dec = nn.Linear(dout,din)
        self.p = torch.rand(din,dout)
        self.enc.weight = nn.Parameter(self.p.t())
        self.dec.weight = self.enc.weight
    
    def encoder(self,x):
        print(x.shape)
        x = self.enc(x)
        x = F.ReLu(x)
        return x

    def decoder(self,x):
        x = F.dec(x.t)
        x = F.sigmoid(x)
        return x

    def forward(self,x):
        return self.decoder(self.encoder(x))


BATCH_SIZE = 32
data_train = DataLoader(MonDataset("mnist_train.csv","label"), shuffle=True, batch_size=BATCH_SIZE)
data_test = DataLoader(MonDataset("mnist_train.csv","label"), shuffle=True, batch_size=BATCH_SIZE)

dimRed = 2
print(next(iter(data_train))[0].shape[1])
autoE = AutoEncoder(next(iter(data_train))[0].shape[1],dimRed)


for x,y in data_train:
    autoE(x)
    exit()
