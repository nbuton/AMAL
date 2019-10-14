import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class RNN(nn.Module):
    def __init__(self, input_size, latent, batch_size):
        super(RNN, self).__init__()
        self.repeated_block = nn.Linear(input_size + latent, latent)
        self.activation = nn.Tanh()
        self.latent_size = latent
        
        self.linear_reg = nn.Linear(latent, input_size)
        

    def forward(self, x, h=None):
        """
        x:      length, batch, dim
        h:              batch, latent
        return: length, batch, dim
        """
        if h is None:
            h = torch.zeros((x.shape[1], self.latent_size))
        
        for l in range(x.shape[0]):
            h = self.one_step(x[l], h)
        
        res = self.linear_reg(h)
        return res
    
    def one_step(self, x, h):
        """
        x:      batch, dim
        h:      batch, latent
        return; batch, latent
        """
        #print("x", x.shape)
        #print("h", h.shape)
        concat = torch.cat((x.double().transpose(0, 1), h.double().transpose(0, 1)))
        concat = concat.transpose(0, 1)
        res = self.repeated_block(concat)
        res = self.activation(res)
        return res

# def train(rnn, loss, optim, temp, cities, epoch,):
    # for e in range(epoch):
    
def main_pred_city_from_temps(temp):
    temp = temp.drop(columns="datetime")
    shape0 = temp.shape
    scaler = StandardScaler()
    temp = scaler.fit_transform(temp.values.reshape(-1,1))
    temp = pd.DataFrame(temp.reshape(shape0))
    temp = temp.fillna(temp.median())
    temp = torch.tensor(temp.values)

    input_size = 30
    latent_size = 100
    batch_size = 8
    test_size = 111
    
    rnn = RNN(input_size, latent_size, batch_size=batch_size).double()
    loss = nn.MSELoss()
    optim = Adam(rnn.parameters(), lr=1e-4)
    
    sequence_size = 200
    iter_max = 1000
    
    # création jeu de test
    test_indice = np.array(range(temp.shape[0] - test_size - sequence_size - 1, temp.shape[0] - sequence_size-1))
    test_x = []
    test_y = []
    for indice,i in enumerate(test_indice):
        test_x.append(temp[i:i+sequence_size])
        test_y.append(temp[i+sequence_size+1])
    test_x = torch.stack(test_x).transpose(0, 1)
    test_y = torch.stack(test_y)
    test_x.requires_grad = False
    test_y.requires_grad = False
    
    list_loss = []
    for e in range(iter_max):
        indices_batch = np.random.randint(0, temp.shape[0] - test_size - 2*sequence_size-1, batch_size)
        batch_x = []
        batch_y = []
        for indice,i in enumerate(indices_batch):
            batch_x.append(temp[i:i+sequence_size])
            batch_y.append(temp[i+sequence_size+1])
        batch_x = torch.stack(batch_x).transpose(0, 1)
        batch_y = torch.stack(batch_y)
        
        test_y_pred = rnn(test_x)
        test_l = loss(test_y_pred, test_y)
        list_loss.append(test_l.item() * np.mean(scaler.scale_) )

        optim.zero_grad()
        y_pred = rnn(batch_x)
        l = loss(y_pred, batch_y)/ sequence_size
        l.backward()
        optim.step()

        """
        if np.isnan(l.item()):
            print("batchx", batch_x)
            print(indices_batch)
            print(indice_ville)
            print(batch_y)
            print(y_pred)
            assert 0
        """
    
    #print(list_loss)
    plt.plot(list_loss)
    plt.show()

def main():
    temp = pd.read_csv("temp_train.csv") 
    main_pred_city_from_temps(temp)

if __name__ == '__main__':
    main()