import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt

class RNN(nn.Module):
    def __init__(self, input_size, latent, n_class, batch_size):
        super(RNN, self).__init__()
        self.repeated_block = nn.Linear(input_size + latent, latent)
        self.activation = nn.Tanh()
        self.default_ini_h = torch.zeros((batch_size, latent))
        assert self.default_ini_h.requires_grad == False
        
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
            h = self.one_step(x[l], h)
        
        res = self.linear_classifier(h)
        return res
    
    def one_step(self, x, h):
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

# def train(rnn, loss, optim, temp, cities, epoch,):
    # for e in range(epoch):
    
def main_pred_city_from_temps(cities, temp):
    target = pd.Categorical(pd.Series(temp.columns[1:]))
    target = pd.get_dummies(target) # one hot encoder
    target = torch.tensor(target.values)
    
    # target = DataLoader(target)
    temp = temp.drop(columns="datetime")
    temp = temp.fillna(273.15)
    # temp = temp.drop(axis=0)
    temp = torch.tensor(temp.values)
    # temp = DataLoader(temp)
    
    # print(target.shape)
    
    input_size = 1
    latent_size = 10
    batch_size = 8
    
    rnn = RNN(input_size, latent_size, batch_size=batch_size, n_class=30).double()
    loss = nn.CrossEntropyLoss()
    optim = Adam(rnn.parameters(), lr=1e-4)
    
    sequence_size = 3
    iter_max = 1000
    
    print(temp.shape)
    print(target.shape)
    list_loss = []
    for e in range(iter_max):
        indices_batch = np.random.randint(0, temp.shape[0] - sequence_size, batch_size)
        indice_ville = np.random.randint(0,len(target),batch_size)
        batch_x = []
        batch_y = []
        for indice,i in enumerate(indices_batch):
            batch_x.append(temp[i:i+sequence_size,indice_ville[indice]])
            batch_y.append(np.argmax(target[indice_ville[indice]]))
        batch_x = torch.stack(batch_x).transpose(0, 1).unsqueeze(2)
        batch_y = torch.stack(batch_y).long()
        
        optim.zero_grad()
        y_pred = rnn(batch_x)
        l = loss(y_pred, batch_y)/ sequence_size
        l.backward()
        optim.step()
        list_loss.append(l.item())
        if np.isnan(l.item()):
            print("batchx", batch_x)
            print(indices_batch)
            print(indice_ville)
            print(batch_y)
            print(y_pred)
            assert 0
    
    print(list_loss)
    plt.plot(list_loss)
    plt.show()

def main():
    cities = pd.read_csv("city_attributes.csv")
    temp = pd.read_csv("temp_train.csv")
    main_pred_city_from_temps(cities, temp)

if __name__ == '__main__':
    main()