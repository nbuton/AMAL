import torch
from torch.autograd import Function
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.datasets
import numpy as np


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
class MSE1(Function):
    @staticmethod
    def forward(ctx, y, yhat):
        assert y.shape == yhat.shape
        ctx.save_for_backward(y, yhat)
        return torch.mean(torch.pow(y - yhat, 2))
    
    @staticmethod
    def backward(ctx, grad_output):
        y, yhat = ctx.saved_tensors
        
        # g_y somme sur sortie[b] de
        #   grad_err_wrt_sortie[b] * grad_sortie[b]_wrt_y[b]
        # = grad_err_wrt_sortie[b] * 2 * (yb - yhatb) / len()
        grad_y     = (y - yhat) * (2 * grad_output / y.nelement()) # type: torch.Tensor
        grad_yhat = - grad_y
        
        assert grad_y.shape == y.shape
        assert grad_yhat.shape == yhat.shape
        return grad_y, grad_yhat


def batch_descent(network, train, loss_f, test=None, batch_size=8, initial_weights=None,
                  initial_bias=None, lr=1e-8, max_iter=1e2, eps=1e-4):
    batch_size = min(batch_size, len(train[0]))
    max_iter = int(max_iter)
    ctx_network = Context()
    ctx_loss = Context()
    _ctx = Context() # non utilisé
    gradient_ini = torch.ones(1, requires_grad=True, dtype=torch.float64)
    
    if initial_weights is None:
        initial_weights = torch.zeros_like(train[0][0], requires_grad=True, dtype=torch.float64)
    assert initial_weights.shape == train[0][0].shape
    
    if initial_bias is None:
        initial_bias = torch.zeros(1, requires_grad=True, dtype=torch.float64)
    assert initial_bias.shape == (1,)
    
    list_loss_train = []
    list_loss_test = []
    def loop(network_weights, network_bias):
        iteration = 0
        while True:
            for i in range(len(train[0]) // batch_size):
                iteration += 1
                batch_x = train[0][i * batch_size: (i + 1) * batch_size]
                batch_y = train[1][i * batch_size: (i + 1) * batch_size]

                pred = network.forward(ctx_network, batch_x, network_weights, network_bias)
                loss = loss_f.forward(ctx_loss, batch_y, pred)
                
                plot_pred = network.forward(_ctx, train[0], network_weights, network_bias)
                plot_loss = loss_f.forward(_ctx, train[1], plot_pred)
                list_loss_train.append(plot_loss.item())
                
                if test is not None:
                    pred_test = network.forward(_ctx, test[0], network_weights, network_bias)
                    list_loss_test.append(loss_f.forward(_ctx, pred_test, test[1]).item())

                if list_loss_train[-1] < eps:
                    return True, iteration, network_weights
                if iteration == max_iter:
                    return False, iteration, network_weights
                
                loss.backward()
                # with torch.no_grad():
                #     # opti.step(); opti.zero_grads()
                #     network_weights -= network_weights.grad * lr
                #     network_bias -= network_bias.grad * lr
                #     network_weights.grad.zero_()
                #     network_bias.grad.zero_()

                _, grad_yhat = MSE1.backward(ctx_loss, gradient_ini)
                grad_x, grad_w, grad_b = Linear1.backward(ctx_network, grad_yhat)# type: torch.Tensor
                with torch.no_grad():
                    assert torch.all(grad_w==network_weights.grad), (grad_w, network_weights.grad)
                    assert torch.all(grad_b==network_bias.grad)

                    network_weights -= lr * grad_w
                    network_bias -= lr * grad_b
                    # remise à zéro automatique quand on se sert d'une variable dans bloc no_grad
                    network_weights.grad.zero_()
                    network_bias.grad.zero_()
                    
                
    
    opt_reached, n_iter, opt = loop(initial_weights, initial_bias)
    if opt_reached:
        print("optimum reached in", n_iter, "last loss=", list_loss_train[-1], list_loss_test[-1])
    else:
        print("optimum not reached in",n_iter, "last loss=", list_loss_train[-1], list_loss_test[-1])
    
    return list_loss_train, list_loss_test

def _main_test_modules():
    data_in = 10
    batch = 5
    
    x = torch.randn(data_in, requires_grad=True, dtype=torch.float64)
    w = torch.randn(data_in, requires_grad=True, dtype=torch.float64)
    b = torch.randn(1, requires_grad=True, dtype=torch.float64)
    
    # Pour tester le gradient
    assert torch.autograd.gradcheck(Linear1Simple.apply, (x, w, b))
    
    x = torch.randn((batch, data_in), requires_grad=True, dtype=torch.float64)
    assert torch.autograd.gradcheck(Linear1.apply, (x, w, b))

    # Pour utiliser la fonction
    # gradient_ini = torch.ones(batch, requires_grad=True, dtype=torch.float64)
    y            = torch.rand(batch, requires_grad=True, dtype=torch.float64)
    yhat         = torch.rand(batch, requires_grad=True, dtype=torch.float64)
    # ctx          = Context()
    # _output = MSE1.forward(ctx, y, yhat)
    # _output_grad = MSE1.backward(ctx, gradient_ini)

    assert torch.autograd.gradcheck(MSE1.apply, (y, yhat))
    print("modules fonctionnels")


def _main_train_module():
    x, y = sklearn.datasets.load_boston(return_X_y=True)
    
    network = Linear1()
    loss_f = MSE1()
    n_splits = 3
    train_size = int(x.shape[0] * (n_splits-1) / n_splits)
    
    for batch_size in [1, 8, train_size]:
        loss_train, loss_test = [], []
        for train_index, test_index in \
                sklearn.model_selection.KFold(n_splits=n_splits, shuffle=False).split(x):
            # todo pourquoi shuffle=True => loss_train == loss_test
            x_train, x_test = torch.tensor(x[train_index], dtype=torch.float64), torch.tensor(x[test_index], dtype=torch.float64)
            y_train, y_test = torch.tensor(y[train_index], dtype=torch.float64), torch.tensor(y[test_index], dtype=torch.float64)
            
            losses = batch_descent(network, (x_train, y_train), loss_f,
                                                  max_iter=1e1,
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


def main():
    _main_test_modules()
    
    _main_train_module()
    
    # Pour telecharger le dataset Boston
    #ds=prepare_dataset("edu.uci.boston")
    #fields, data =ds.files.data()


if __name__ == '__main__':
    main()