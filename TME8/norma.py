import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.datasets import MNIST
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torch.nn.functional as F
from datetime import datetime


torch.autograd.set_detect_anomaly(True)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

def l1_reg(factor=1.):
    def f(net):
        L1_reg = torch.tensor(0.)
        for name, param in net.named_parameters():
            # if hasattr("named_parameters", param):
            if 'weight' in name:
                L1_reg += torch.norm(param, 1)
        
        return L1_reg * factor
    
    return f


class FC(nn.Module):
    def __init__(self, batchnorm=False, layernorm=False, dropout=0.):
        super().__init__()
        l1, l2 = [], []
        l1.append(nn.Linear(28 * 28, 100))
        l2.append(nn.Linear(100, 100))
        if batchnorm:
            l1.append(nn.BatchNorm1d(100))
            l2.append(nn.BatchNorm1d(100))
        if layernorm:
            l1.append(nn.LayerNorm(100))
            l2.append(nn.LayerNorm(100))
        l1.append(nn.ReLU())
        l2.append(nn.ReLU())
        if dropout:
            l1.append(nn.Dropout(dropout))
            l2.append(nn.Dropout(dropout))
        
        self.fc1 = nn.Sequential(*l1)
        self.fc2 = nn.Sequential(*l2)
        self.fc3 = nn.Linear(100, 10)
    
    def forward(self, x0):
        batch_size = x0.shape[0]
        x0 = x0.reshape((batch_size, -1))
        x1 = self.fc1(x0)
        x2 = self.fc2(x1)
        x3 = self.fc3(x2)
        
        return x1, x2, x3


def store_grad(var):
    def hook(grad):
        var.grad = grad
    
    return var.register_hook(hook)  #appelé à la fin de backward()


def fit_model(model, data_train, data_test, writer, n_epochs=100):
    for epoch in range(n_epochs):
        if epoch % (n_epochs // 10) == 0 and epoch:
            print(epoch, "/", n_epochs)
        
        for x, y in data_train:
            x.requires_grad = True
            model["net"].zero_grad()
            hid1, hid2, y_pred = model["net"](x)
            
            loss = model["loss"](y_pred, y)
            if model["weight_loss"] is not None:
                loss += model["weight_loss"](model)
            
            store_grad(x)
            hid1.retain_grad()
            hid2.retain_grad()
            loss.backward()
            model["optim"].step()
        
        print(loss)
        print(y_pred.min(), y_pred.max())
        print(x.grad.abs().max(), x.grad.abs().mean())
        print(hid1.grad.abs().max(), hid1.grad.abs().mean())
        writer.add_scalar("loss", loss.item(), epoch)
        # entropy(writer, y_pred.detach(), epoch)
        hist_grad(writer, x, hid1, hid2, epoch)
        hist_weights(writer, model["net"], epoch)
        hist_sorties(writer, hid1, hid2, y_pred, epoch)


def entropy(writer, sortie, epoch):
    all_sum = []
    for one_sortie in sortie:
        one_sortie = F.softmax(one_sortie, dim=0)
        somme = 0
        for pi in one_sortie:
            somme += pi * np.log(pi)
        all_sum.append(-somme)
    writer.add_histogram('entropy', torch.tensor(all_sum), epoch)


def hist_grad(writer, x, hid1, hid2, epoch):
    writer.add_histogram('g1', x.grad, epoch)
    writer.add_histogram('g2', hid1.grad, epoch)
    writer.add_histogram('g3', hid2.grad, epoch)


def hist_weights(writer, net, epoch):
    writer.add_histogram('w1', net.fc1[0].weight, epoch)
    writer.add_histogram('w2', net.fc2[0].weight, epoch)
    writer.add_histogram('w3', net.fc3.weight, epoch)


def hist_sorties(writer, hid1, hid2, y_pred, epoch):
    writer.add_histogram('o1', hid1, epoch)
    writer.add_histogram('o2', hid2, epoch)
    writer.add_histogram('o3', y_pred, epoch)


def main():
    nets = [
        # ("Vanilla", FC(), None),
        ("L1", FC(), ('L1,', 1e6)),
        ("L2", FC(), ('L2,', 1e6)),
        # ("BN", FC(batchnorm=True), None),
        # ("LN", FC(layernorm=True), None),
        # ("Dropout", FC(dropout=.6), None)
    ]
    
    ds = MNIST("~/datasets/MNIST", download=True,
               transform=transforms.Compose([
                   #transforms.RandomCrop(28),
                   #transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   #transforms.Normalize((.491, .482, .447), (.202, .199, .201)), #mean std
               ]))
    sizes = [int(.05 * len(ds)), int(.2 * len(ds)), len(ds) - int(.05 * len(ds)) - int(.2 * len(ds))]
    dataset_train, dataset_test, _ = torch.utils.data.random_split(ds, sizes)
    
    data_train = DataLoader(dataset_train, batch_size=300)
    data_test = DataLoader(dataset_test, batch_size=300)
    
    t = str(datetime.now().strftime('%b%d_%H-%M-%S'))
    
    for name, net, reg in nets:  #type: _, nn.Module, _
        print(name)
        writer = SummaryWriter("histo/" + t + "/" + name)
        
        if reg is not None and reg[0] == "L1":
            weight_loss = l1_reg()
        else:
            weight_loss = None
        
        if reg is not None and reg[0] == "L2":
            optim = Adam(net.parameters(), lr=1e1, weight_decay=reg[1])
        else:
            optim = Adam(net.parameters(), lr=1e1)
        
        fit_model({
            "net": net,
            "loss": nn.CrossEntropyLoss(),
            "optim": optim,
            "weight_loss": weight_loss,
        }, data_train, data_test, writer=writer, n_epochs=10)
        
        writer.close()


main()
