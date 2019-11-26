import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, random_split

from torch.utils.tensorboard import SummaryWriter 

import torchvision.transforms as transforms
import torch.nn.functional as F

class FC(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 10)
        self.relu = nn.ReLU()

    def forward(self, x0):
        batch_size = x0.shape[0]
        x0 = x0.reshape((batch_size, -1))
        x1 = self.fc1(x0)
        x1 = self.relu(x1)
        x2 = self.fc2(x1)
        x2 = self.relu(x2)
        x3 = self.fc3(x2)

        return x1, x2, x3

def store_grad(var):
    def hook(grad):
        var.grad = grad
    return var.register_hook(hook) #appelé à la fin de backward()


def fit_model(model, data_train, data_test, writer, n_epochs=10):
    for epoch in range(n_epochs):
        hid1, hid2, y_pred = None, None, None
        for x, y in data_train:
            x.requires_grad = True
            model["net"].zero_grad()
            hid1, hid2, y_pred = model["net"](x)
            loss = model["loss"](y_pred, y)
            store_grad(x)
            hid1.retain_grad()
            hid2.retain_grad()
            loss.backward()
            model["optim"].step()
        entropy(writer,y_pred,epoch)
        hist_grad(writer,model,x,hid1,hid2,epoch)
        hist_weights(writer, model["net"], epoch)
        hist_sorties(writer, hid1, hid2, y_pred, epoch)
    writer.close()

def entropy(writer, sortie,epoch):
    all_sum = []
    for one_sortie in sortie:
        one_sortie = F.softmax(one_sortie, dim=0)
        somme = 0
        for pi in one_sortie:
            somme+=pi*np.log(pi)
        all_sum.append(-somme)
    print(all_sum)
    writer.add_histogram('entropy', torch.tensor(all_sum) , epoch)
    


def hist_grad(writer, net,x,hid1,hid2,epoch):
    writer.add_histogram('e1', x.grad, epoch)
    writer.add_histogram('e2', hid1.grad, epoch)
    writer.add_histogram('e3', hid2.grad, epoch)


def hist_weights(writer, net, epoch):
    writer.add_histogram('fc1', net.fc1.weight, epoch)
    writer.add_histogram('fc2', net.fc2.weight, epoch)
    writer.add_histogram('fc3', net.fc3.weight, epoch)

def hist_sorties(writer, hid1, hid2, y_pred, epoch):
    writer.add_histogram('s1', hid1, epoch)
    writer.add_histogram('s2', hid2, epoch)
    writer.add_histogram('s3', y_pred, epoch)


def main():
    writer = SummaryWriter("histo/vanilla")#/L1

    net = FC()
    ds = MNIST("~/datasets/MNIST", download=True,
        transform=transforms.Compose([
            #transforms.RandomCrop(28),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize((.491, .482, .447), (.202, .199, .201)), #mean std
]))
    sizes = [int(.05*len(ds)), int(.2*len(ds)), len(ds) - int(.05*len(ds)) -int(.2*len(ds)) ]
    dataset_train, dataset_test, _ = torch.utils.data.random_split(ds, sizes)

    data_train = DataLoader(dataset_train, batch_size=300)
    data_test = DataLoader(dataset_test, batch_size=300)
    fit_model({
        "net":  net,
        "loss": nn.CrossEntropyLoss(),
        "optim": Adam(net.parameters(),lr=1e-4)
    }, data_train, data_test,
    writer=writer)

main()