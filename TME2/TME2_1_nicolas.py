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
class Linear1Simple(Function):
    """retourne un scalaire, ne gère pas les batch"""

    @staticmethod
    def forward(ctx, x, w, b):
        """calcul de la sortie du module"""
        assert x.ndimension() == 1, "x: (data_in, )"
        data_in, = x.shape
        assert w.ndimension() == 1, "w: (data_in, )"
        assert w.shape[0] == data_in
        assert b.shape == (1,)

        # sauvegarde des éléments utiles au backward
        ctx.save_for_backward(x, w)

        sortie = torch.dot(x, w) + b
        assert sortie.shape == (1,)
        return sortie

    @staticmethod
    def backward(ctx, grad_output):
        """calcul du gradient de l'erreur par rapport à chaque groupe d'entrées du module
        en se servant du gradient de la sortie du module"""
        # chargement des éléments utiles
        x, w = ctx.saved_tensors
        assert grad_output.shape == (1,), "grad_output doit avoir la même shape que output"

        grad_x = grad_output * w
        grad_w = grad_output * x
        grad_b = grad_output
        return grad_x, grad_w, grad_b


# noinspection PyMethodOverriding
class Linear1(Function):
    """retourne un scalaire"""

    @staticmethod
    def forward(ctx, x, w, b):
        """calcul de la sortie du module"""
        assert x.ndimension() == 2, "x: (batch, data_in)"
        batch, data_in = x.shape
        assert w.ndimension() == 1, "w: (data_in, )"
        assert w.shape[0] == data_in
        assert b.shape == (1,)

        # sauvegarde des éléments utiles au backward
        ctx.save_for_backward(x, w)

        # b se broadcast sur la sortie
        sortie = torch.matmul(x, w) + b
        assert sortie.shape == (batch,)
        return sortie

    @staticmethod
    def backward(ctx, grad_output):
        """calcul du gradient de l'erreur par rapport à chaque groupe d'entrées du module
        en se servant du gradient de la sortie du module
        x      : batch, in
        w      : in
        b      : 1
        grad_o : batch
        leurs gradients ont la même taille """
        x, w = ctx.saved_tensors
        batch, data_in = x.shape
        assert grad_output.shape == (batch,), "grad_output doit avoir la même shape que output"

        # g_x[b,i]
        #   grad_err_wrt_sortie[b] * grad_sortie[b]_wrt_x[b,i]
        # = grad_err_wrt_sortie[b] * wi
        grad_x = torch.matmul(grad_output.unsqueeze(1), w.unsqueeze(0))

        # g_w[i] = SOMME sur les sorties[b] de
        #   grad_err_wrt_sortie[b] * grad_sortie[b]_wrt_w[i]
        # = grad_err_wrt_sortie[b] * x[b][i]
        grad_w = torch.matmul(grad_output, x)


        # g_bias = somme sur les sorties[b] de
        #   grad_err_wrt_sortie[b] * grad_sortie_wrt_bias
        # = grad_err_wrt_sortie[b] * 1
        grad_b = torch.sum(grad_output).unsqueeze(0)

        assert grad_x.shape == x.shape
        assert grad_w.shape == w.shape
        assert grad_b.shape == (1,), grad_b.shape
        return grad_x, grad_w, grad_b


# noinspection PyMethodOverriding
class MSE(Function):
    @staticmethod
    def forward(ctx, y, yhat):
        ctx.save_for_backward(y,yhat)
        return torch.sum(torch.pow(y-yhat,2))

    @staticmethod
    def backward(ctx, grad_output):
        """
        Taille :
        y : batch
        yhat : batch
        grad_output : batch
        """
        y,yhat = ctx.saved_tensors
        #Element wise multiplication
        derive_y = torch.mul(2*(y-yhat),grad_output)
        derive_yhat = torch.mul(2*(yhat-y),grad_output)
        assert derive_y.shape == y.shape
        assert derive_yhat.shape == yhat.shape
        return derive_y,derive_yhat


def main():
    data_in = 10
    batch = 5

    ## Pour telecharger le dataset Boston
    data = pd.read_csv("BostonHousing.csv")
    #print(data.describe())
    data = data.sample(frac=1).reset_index(drop=True)
    prop_train = 0.8
    all_data_target = torch.tensor(data['medv'].values.astype(np.float64))
    all_data = torch.tensor(data.drop('medv', axis = 1).values.astype(np.float64))
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
    w = torch.randn(data_in, requires_grad=True, dtype=torch.float64)
    b = torch.randn(1,       requires_grad=True, dtype=torch.float64)
    for n_iter in range(nb_iter):
        w.requires_grad = True
        b.requires_grad = True
        choix = np.random.randint(0,len(train))
        #On rajoute la taille de batch de 1
        x = train[choix].unsqueeze(0)
        y = train_target[choix].unsqueeze(0)
        pred = Linear1.apply(x,w,b)
        erreur = MSE.apply(y,pred)
        #On calcul les backwards
        gradient_originel = torch.ones((), dtype=torch.float64)
        erreur.backward()
        grad_w,grad_b = w.grad,b.grad
        with torch.no_grad():
            w = w - (epsilon*(grad_w/x.shape[0]))
            b = b - (epsilon*(grad_b/x.shape[0]))
        #erreur.grad_zeros()
        loss_train = np.mean(MSE.apply(train_target,Linear1.apply(train,w,b)).detach().numpy())
        writer.add_scalar('Loss_stochastique/train', loss_train, n_iter)
        loss_test = np.mean(MSE.apply(test_target,Linear1.apply(test,w,b)).detach().numpy())
        writer.add_scalar('Loss_stochastique/test', loss_test, n_iter)


if __name__ == '__main__':
    main()
