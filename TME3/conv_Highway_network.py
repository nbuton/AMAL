import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


class MonDataset(Dataset):
    def __init__(self,nameCsv,nameLabel):
        data = pd.read_csv(nameCsv)
        data = data.sample(frac=1).reset_index(drop=True)
        self.all_data_target = torch.Tensor(data[nameLabel].values.astype(np.int8))
        self.all_data = torch.Tensor(data.drop(nameLabel, axis = 1).values.astype(np.float64))
        self.all_data = self.all_data/255
        self.all_data = self.all_data.reshape((self.all_data.shape[0],1,28,28))

    def __getitem__(self,index):
        return self.all_data[index],self.all_data_target[index]

    def __len__(self):
        return len(self.all_data)

class Layer_H(nn.Module):
    def __init__(self,din):
        super(Layer_H,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.T = nn.Linear(din,1)

    def forward(self,x):
        t = self.T(x.reshape(x.shape[0],784))
        #print(self.conv1(x).shape)
        #print(t.shape)
        #print((1-t).shape)
        #print(x.shape)
        x_new_shape = x.reshape(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3])
        #print(x_new_shape*(1-t))
        prop_x = x_new_shape*(1-t)
        prop_x = prop_x.reshape(x.shape)
        conv_new_shape = F.relu(self.conv1(x))
        conv_new_shape = conv_new_shape.reshape(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3])
        prop_conv = conv_new_shape*t
        prop_conv = prop_conv.reshape(x.shape)
        x = prop_conv + prop_x
        #print("#########")
        #print(x.shape)
        #print("&&&&&&")
        return x

class Highway(nn.Module):
    def __init__(self,n_layer,d_in):
        super(Highway,self).__init__()
        self.n_layer = n_layer
        self.all_layer = []
        for k in range(n_layer):
            self.all_layer.append(Layer_H(d_in))
        self.final_fc = nn.Linear(784,10)

    def forward(self,x):
        for k in range(self.n_layer):
            x = self.all_layer[k](x)
        x = self.final_fc(x.reshape(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3]))
        return F.sigmoid(x)



class State:
    def __init__(self,model,optim):
        self.model=model
        self.optim=optim
        self.epoch,self.iteration=0,0


#permet de selectionner le gpu si disponible
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
savepath = my_file = Path("HighwayModel.pkl")

if savepath.is_file():
    #on recommence depuis le modele sauvegarde
    with savepath.open("rb") as fp:
        state=torch.load(fp)
else:
    dimEntree = 784
    n_layer = 4
    learning_rate = 1e-6
    high=Highway(n_layer,dimEntree)
    high=high.to(device)
    optim = torch.optim.Adam(high.parameters(), lr=learning_rate)
    state=State(high,optim)


ITERATIONS = 5
BATCH_SIZE = 100

data_train = DataLoader(MonDataset("mnist_train.csv","label"), shuffle=True, batch_size=BATCH_SIZE)
data_test = DataLoader(MonDataset("mnist_train.csv","label"), shuffle=True, batch_size=BATCH_SIZE)

#Nombre d'étape de moyennation de l'erreur pour l'enregister dans tensorboard
n_moy = 10
writer = SummaryWriter("courbe_Highway_50")
loss_moyen = 0
it_for_loss = 0

loss = nn.CrossEntropyLoss()


for epoch in range(state.epoch,ITERATIONS):
    print("epoch",epoch,"/",state.epoch)
    for x,y in data_train:
        state.optim.zero_grad()
        #On transfert les données sur GPU si disponible
        x = x.to(device)
        xhat=state.model(x)
        y = y.view(-1).long()
        l=loss(xhat,y)
        l.backward(retain_graph=True)
        state.optim.step()
        loss_moyen+=l
        state.iteration+=1
        it_for_loss+=1
        if(it_for_loss%n_moy==0):
            loss_train = loss_moyen/n_moy
            writer.add_scalar('Loss_stochastique/train', loss_train, state.iteration)
            loss_moyen = 0
            if(it_for_loss%(n_moy*10)==0):
                print(state.iteration)
                print(loss_train)


    with savepath.open("wb") as fp:
        state.epoch=epoch+1
        torch.save(state,fp)
