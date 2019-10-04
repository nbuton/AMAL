# Ne pas oublier d'executer dans le shell avant de lancer python :
# source /users/Enseignants/piwowarski/venv/amal/3.7/bin/activate

import torch
from torch.autograd import Function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import numpy as np
#from torch.autograd import gradcheck
#from datamaestro import prepare_dataset


class Context:
    """Very simplified context object"""
    def __init__(self):
        self._saved_tensors = ()
    def save_for_backward(self, *args):
        self._saved_tensors = args
    @property
    def saved_tensors(self):
        return self._saved_tensors



# noinspection PyMethodOverriding
class Linear1(torch.nn.Module):
    """retourne un scalaire"""
    def __init__(self,datain):
        super(Linear1, self).__init__()
        self.fc = torch.nn.Linear(datain, 1)

    def forward(self, x):
        return self.fc(x)


# noinspection PyMethodOverriding
class MSE(torch.nn.Module):
    """retourne un scalaire"""
    def __init__(self):
        super(MSE, self).__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, y, y_pred):
        return self.mse(y,y_pred)


def main():
    data_in = 10
    batch = 5

    ## Pour telecharger le dataset Boston
    data = pd.read_csv("BostonHousing.csv")
    data = data.sample(frac=1).reset_index(drop=True)
    prop_train = 0.8
    all_data_target = torch.tensor(data['medv'].values.astype(np.float32))
    all_data = torch.tensor(data.drop('medv', axis = 1).values.astype(np.float32))
    nb_train = int(prop_train*len(all_data))
    train_target = all_data_target[:nb_train]
    train = all_data[:nb_train]
    test_target = all_data_target[nb_train:]
    test = all_data[nb_train:]

    writer = SummaryWriter("courbe_SGD")

    #Descente stochastique

    nb_iter = 100
    epsilon = 1e-6
    data_in = train.shape[1]
    linear = Linear1(data_in)
    mse = MSE()
    optimizer = torch.optim.SGD(linear.parameters(), lr=1e-7)
    for n_iter in range(nb_iter):
        optimizer.zero_grad()
        choix = np.random.randint(0,len(train))
        #On rajoute la taille de batch de 1
        x = train[choix].unsqueeze(0)
        y = train_target[choix].unsqueeze(0)
        pred = linear(x)
        erreur = mse(y,pred)
        #On calcul les backwards
        erreur.backward()
        optimizer.step()
        loss_train = mse(train_target,linear(train)[:,0]).detach().numpy()
        writer.add_scalar('Loss_stochastique/train', loss_train, n_iter)
        loss_test = mse(test_target,linear(test)[:,0]).detach().numpy()
        writer.add_scalar('Loss_stochastique/test', loss_test, n_iter)


if __name__ == '__main__':
    main()
