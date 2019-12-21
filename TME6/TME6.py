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



class LSTM_cell(nn.Module):
    def __init__(self, taille_embedding, taille_latent):
        super(Sentiment, self).__init__()
        self.activationT = nn.Tanh()
        self.activationS = nn.sigmoid()
        self.linearF1 = nn.Linear(latent, n_class)
        self.linearF2 = nn.Linear(latent, n_class)
        self.linearI1 = nn.Linear(latent, n_class)
        self.linearI2 = nn.Linear(latent, n_class)
        self.linearM1 = nn.Linear(latent, n_class)
        self.linearM2 = nn.Linear(latent, n_class)
        self.linearO1 = nn.Linear(latent, n_class)
        self.linearO2 = nn.Linear(latent, n_class)

    def forward(self, c_h_t_1 , x_t):
        c_t_1, h_t_1 = c_h_t_1
        f_t = self.activationS(self.linearF1(h_t_1) + self.linearF2(x_t))
        i_t = self.activationS(self.linearI1(h_t_1) + self.linearI2(x_t))
        c_t = f_t * c_t_1 + i_t * self.activationT(self.linearM1(h_t_1)+self.linearM2(x_t))
        o_t = self.activationS(self.linearO1(h_t_1)+self.linearO2(x_t))
        h_t = o_t * self.activationT(c_t)
        return [c_t, h_t]

class LSTM_cell_v2(nn.Module):
    def __init__(self, taille_embedding, taille_latent):
        super(Sentiment, self).__init__()
        self.activationT = nn.Tanh()
        self.activationS = nn.sigmoid()
        self.linearH = nn.Linear(taille_latent, taille_latent*4)
        self.linearI = nn.Linear(taille_embedding, taille_latent*4)

    def forward(self, c_h_t_1, x_t):
        c_t_1, h_t_1 = c_h_t_1
        tE = self.taille_embedding
        comb_x , comb_h = self.linearH(h_t_1) + self.linearI(x_t)
        f_t = self.activationS(comb_h[:tE]+comb_x[:tE])
        i_t = self.activationS(comb_h[tE:tE*2]+comb_x[tE:tE*2])
        c_t = f_t * c_t_1 + i_t * self.activationT(comb_h[tE*2:tE*3]+comb_x[tE*2:tE*3])
        o_t = self.activationS(comb_h[tE*3:]+comb_x[tE*3:])
        h_t = o_t * self.activationT(c_t)
        return [c_t, h_t]

class RNN_cell(nn.Module):
    def __init__(self, taille_embedding , taille_latent):
        super(Sentiment, self).__init__()
        self.repeated_block = nn.Linear(input_size + latent, latent)
        self.activation = nn.Tanh()

    def forward(self, c_h_t_1, x_t):
        """
        x:      batch, dim
        h:      batch, latent
        return; batch, latent
        """
        # print("x", x.shape)
        # print("h", h.shape)
        concat = torch.cat((x.double().transpose(0, 1), h.double().transpose(0, 1)))
        concat = concat.transpose(0, 1)
        res = self.repeated_block(concat)
        res = self.activation(res)
        return res


class ReccurentNN(nn.Module):
    def __init__(self, input_size, taille_latent, n_class, batch_size, cell_type):
        super(RNN, self).__init__()
        self.embed = nn.Embedding(taille_dico,taille_embedding)
        if(cell_type == "rnn"):
            self.cell = RNN_cell(taille_embedding, taille_latent)
        elif(cell_type == "lstm"):
            self.cell = LSTM_cell_v2(taille_embedding, taille_latent)
        elif(cell_type == "gru"):
            self.cell = GRU_cell(taille_embedding, taille_latent)
        else:
            print("Type de cellule non connue")
            exit(1)
        self.default_ini_h = torch.zeros((batch_size, latent))
        self.linear_classifier = nn.Linear(latent, n_class)
        

    def forward(self, x, h=None):
        """
        x:      length, batch, dim
        h:              batch, latent
        return: length, batch, latent
        """
        if h is None:
            h = self.default_ini_h
        
        for l in range(x.shape[0]):
            h = self.cell(x[l], h)
        
        res = self.linear_classifier(h)
        return res
    


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