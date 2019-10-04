import torch
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.datasets
import numpy as np
import torch.nn as nn
import torch.optim

class Reseau1(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(13, 1)
        with torch.no_grad():
            self.lin.weight.zero_()
            self.lin.bias.zero_()
        self.tanh = nn.Tanh()
        self.mse = nn.MSELoss()
    
    def forward(self, x, y):
        x = self.lin(x)
        x = self.tanh(x)
        res = self.mse(x, y)
        return res
    
    def params(self):
        return [self.lin.weight, self.lin.bias]


class Optimiser1(torch.optim.Optimizer):
    def __init__(self, list_params, lr):
        super(Optimiser1, self).__init__(list_params, {})
        self.list_params = list_params
        self.lr = lr

    def step(self):
        """remet les gradients à zéro"""
        with torch.no_grad():
            for p in self.list_params:
                old = p.grad.clone() # type: torch.Tensor
                assert torch.all(old==p.grad)
                p -= p.grad * self.lr
    
    def zero_grads(self):
        for p in self.list_params:
            p.grad.zero_()


def batch_descent(network, opti, train, test=None, batch_size=8,
                  max_iter=1e2, eps=1e-4):
    batch_size = min(batch_size, len(train[0]))
    max_iter = int(max_iter)

    list_loss_train = []
    list_loss_test = []

    def loop():
        iteration = 0
        while True:
            for i in range(len(train[0]) // batch_size):
                iteration += 1
                batch_x = train[0][i * batch_size: (i + 1) * batch_size]
                batch_y = train[1][i * batch_size: (i + 1) * batch_size]

                pred = network.forward(batch_x, batch_y)
                
                plot_pred = network.forward(train[0], train[1])
                list_loss_train.append(plot_pred.item())
                
                if test is not None:
                    pred_test = network.forward(test[0], test[1])
                    list_loss_test.append(pred_test.item())

                if list_loss_train[-1] < eps:
                    return True, iteration
                if iteration == max_iter:
                    return False, iteration
                
                pred.backward()

                opti.step()
                opti.zero_grads()
                    
                
    
    opt_reached, n_iter = loop()
    if opt_reached:
        print("optimum reached in", n_iter, "last loss=", list_loss_train[-1], list_loss_test[-1])
    else:
        print("optimum not reached in",n_iter, "last loss=", list_loss_train[-1], list_loss_test[-1])
    
    return list_loss_train, list_loss_test


def main():
    x, y = sklearn.datasets.load_boston(return_X_y=True)
    y = y.reshape((-1, 1))
    n_splits = 3
    train_size = int(x.shape[0] * (n_splits-1) / n_splits)
    
    for batch_size in [1, 8, train_size]:
        loss_train, loss_test = [], []
        for train_index, test_index in \
                sklearn.model_selection.KFold(n_splits=n_splits, shuffle=False).split(x):
            x_train, x_test = torch.tensor(x[train_index], dtype=torch.float64), torch.tensor(x[test_index], dtype=torch.float64)
            y_train, y_test = torch.tensor(y[train_index], dtype=torch.float64), torch.tensor(y[test_index], dtype=torch.float64)
            
            r = Reseau1().double()
            o = Optimiser1(r.params(), lr=1e-8)
            
            losses = batch_descent(r, o, (x_train, y_train), max_iter=1e1,
                            test=(x_test, y_test), batch_size=batch_size)
            loss_train.append(losses[0])
            loss_test.append(losses[1])
        
        loss_train = np.array(loss_train)
        loss_test = np.array(loss_test)
        
        plt.figure()
        plt.plot(np.mean(loss_train, axis=0), label="train")
        plt.plot(np.mean(loss_test , axis=0), label="test")
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.legend()
        plt.title("batchsize=" + str(batch_size))
    plt.show()


main()