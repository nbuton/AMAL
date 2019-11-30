import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ConvNetwork(nn.Module):
    #networkDesc est une liste de couple avec comme premier element le nom du paramettre et le deuxieme l'indice dans le dict
    #Exemple : [("Number_of_filters",0),("Filter_size",3),("Stride",2)]
    def __init__(self,networkDesc,dict_params):
        super(ConvNetwork, self).__init__()
        i = 0
        self.out_channels = dict_params[networkDesc[i][0]][networkDesc[i][1]]
        i+=1
        self.kernel_size = dict_params[networkDesc[0][0]][networkDesc[0][1]]
        i+=1
        self.stride = dict_params[networkDesc[0][0]][networkDesc[0][1]]

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.out_channels , kernel_size=self.kernel_size, stride=self.stride)
        self.pool = nn.MaxPool2d(2, 2)
        self.flat = nn.Flatten(end_dim=-1)
        #Peut etre a revoir la taille d'entrée
        self.fc1 = nn.Linear(self.out_channels*(32//self.stride), 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, self.out_channels*(32//self.stride))
        x = self.fc1(x)
        return x

class ControllerNetwork(nn.Module):
    #list_desc example : list_desc=["Number_of_filters","Filter_size","Stride"]
    def __init__(self,list_desc,dict_params,embedding_size, hidden_size,num_layers):
        super(ControllerNetwork, self).__init__()
        self.list_desc = list_desc
        self.dict_params = dict_params
        self.taille = len(list_desc)
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        #Le input size est de la taille de l'embedding
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers)
        self.list_embed_weights = []
        for desc in self.list_desc:
            self.list_embed_weights.append(nn.Linear(hidden_size,embedding_size))
        self.soft = nn.Softmax()


    def forward(self, inutile):
        h = torch.zeros([self.hidden_sizes], dtype=torch.int32)
        c = torch.zeros([self.hidden_sizes], dtype=torch.int32)
        x = torch.zeros([1,self.embedding_size], dtype=torch.int32)
        all_decision = torch.tensor([])
        for embed in self.list_embed_weights:
            h_n,c_n = self.rnn(x, (h, c))
            h = h_n[-1]
            c = c_n[-1]
            decision = self.soft(embed(h))
            all_decision = torch.concat(all_decision, decision)
            x = torch.matmul(embed.weight.t(), decision)
        return all_decision



class NetworkTester():
    def __init__(self):
        #Définition des différentes transformations que l'on effectue sur les images
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        #Chargement de la base CIFAR10 pour le train,validation et test
        allset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        split = int(0.8*len(allset))
        trainset = allset[split:]
        self.valset = allset[:split]
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
        self.valloader = torch.utils.data.DataLoader(valset, batch_size=4, shuffle=True, num_workers=2)

        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

        criterion = nn.CrossEntropyLoss()

    def get_reward(self,net):
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        for epoch in range(2):
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
        return accuracy(net(self.valset.train),self.valset.test)

    def test_score(self,net):
        return accuracy(net(self.testset.train),self.testset.test)


def update_controler(controler,reward, log_probs):
    nb_hyperparams = len(log_probs)
    rewards = [reward]*nb_hyperparams
    discounted_rewards = []

    for t in range(len(rewards)):
        Gt = 0
        pw = 0
        for r in rewards[t:]:
            Gt = Gt + GAMMA**pw * r
            pw = pw + 1
        discounted_rewards.append(Gt)

    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards

    policy_gradient = []
    for log_prob, Gt in zip(log_probs, discounted_rewards):
        policy_gradient.append(-log_prob * Gt)

    controler.optimizer.zero_grad()
    policy_gradient = torch.stack(policy_gradient).sum()
    policy_gradient.backward()
    controler.optimizer.step()


dict_params = dict()
dict_params["Number_of_filters"] = [24, 36, 48, 64]
dict_params["Filter_size"] = [1, 3, 5, 7]
dict_params["Stride"] = [1, 2, 3, 4, 5]
MAX_ITER = 10

controler = ControllerNetwork()
netTester = NetworkTester()
best_model = None
Rmax = -99999999

for k in range(MAX_ITER):
    log_probs = controler(torch.tensor([]))
    list_desc = argmax(log_probs)
    model_to_test = ConvNetwork(list_desc)
    R = netTester.get_reward(model_to_test)
    if(R>Rmax):
        best_model = model_to_test
        Rmax = R
    update_controler(controler,R,log_probs)

print(netTester.test_score(best_model))
