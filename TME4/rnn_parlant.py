import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

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
    
def main_pred_city_from_temps(letter_vec):
    temp = np.array(letter_vec)

    input_size = 1
    latent_size = 10
    batch_size = 8
    
    rnn = RNN(input_size, latent_size, batch_size=batch_size, n_class=30).double()
    loss = nn.CrossEntropyLoss()
    optim = Adam(rnn.parameters(), lr=1e-4)
    
    sequence_size = 10
    iter_max = 10000


    list_loss = []
    for e in range(iter_max):
        indices_batch = np.random.randint(0, temp.shape[0] - sequence_size-1, batch_size)
        batch_x = []
        batch_y = []
        for indice,i in enumerate(indices_batch):
            batch_x.append(temp[i:i+sequence_size])
            batch_y.append(temp[i+sequence_size+1])
        batch_x = torch.stack(batch_x).transpose(0, 1).unsqueeze(2)
        batch_y = torch.stack(batch_y).long()
        
        test_y_pred = rnn(test_x)
        test_l = loss(test_y_pred, test_y)
        list_loss.append(test_l.item())

        optim.zero_grad()
        y_pred = rnn(batch_x)
        l = loss(y_pred, batch_y)/ sequence_size
        l.backward()
        optim.step()

    plt.plot(list_loss)
    plt.show()

def main():
    all_txt = " ".join(open("descarte.txt").readlines())
    all_txt_vect = [ord(l) for l in all_txt]
    print(len(all_txt_vect))
    main_pred_city_from_temps(all_txt_vect)

if __name__ == '__main__':
    main()