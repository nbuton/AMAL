import re
from pathlib import Path
from torch.utils.data import Dataset
from datamaestro import prepare_dataset
from gensim.test.utils import datapath, get_tmpfile
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import numpy as np
import torch.nn.utils.rnn as utilsRNN
import pickle
import torch.utils.data as data_utils
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import matplotlib.pyplot as plt

class Reseaux_ex_0(nn.Module):
    def __init__(self, taille_embedding, glove_weight):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(glove_weight)
        self.fc = nn.Linear(taille_embedding,2)


    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x,dim=1)
        x = self.fc(x)
        return x

class Reseaux_ex_1(nn.Module):
    def __init__(self, taille_embedding, glove_weight):
        super().__init__()
        self.taille_embedding = taille_embedding
        self.embedding = nn.Embedding.from_pretrained(glove_weight)
        self.fc = nn.Linear(taille_embedding,2)
        self.q = torch.nn.Parameter(torch.ones(taille_embedding))
        self.bias = torch.nn.Parameter(torch.zeros(taille_embedding))
        torch.nn.init.uniform_(self.q, a=-0.0, b=1.0)

    def forward(self, x):
        taille_seq = x.shape[1]
        batch_size = x.shape[0]
        x = self.embedding(x)
        q_exp = self.q.expand((batch_size,taille_seq,self.taille_embedding))
        #print(x.shape)
        #print(q_exp.shape)
        #print(self.bias.shape)
        P_a_t = torch.nn.functional.softmax(torch.mul(q_exp,x) + self.bias)
        #print(P_a_t.shape)
        #print(x.shape)
        x = torch.mul(P_a_t,x)
        x = torch.sum(x,dim=1)
        #print(x.shape)
        x = self.fc(x)
        return x


def collate(batch):
    data = [item[0] for item in batch]
    labels = torch.tensor([[0.0,1.0] if item[1]==1 else [1.0,0.0] for item in batch])
    new_data = pad_sequence(data)
    new_data = new_data.transpose(0,1)
    return new_data,labels


#Data est une liste de taille 50 000 contenant des listes représentant les phrases et dans ces meme liste nous avons les indices des mots
#Target est un tenseur de taille 50 000 avec tout les labels des differentes review
with open('imdb_train.pickle', 'rb') as handle:
    train = pickle.load(handle)

#Matrice glove avec correspondance avec les indices des mots dans la data
with open('matrice_glove.pickle', 'rb') as handle:
    matrice_glove = pickle.load(handle)

print(max([len(phrase) for phrase in train]))

#A faire par batch
#pad_sequence(data).size()

#train_tensor = data_utils.TensorDataset(data, target) 
batch_size=32
train_loader = data_utils.DataLoader(dataset = train, batch_size = batch_size, shuffle = True,collate_fn=collate)

EMBEDDING_SIZE = 50
reseau_0 = Reseaux_ex_1(EMBEDDING_SIZE,matrice_glove)
optimizer = torch.optim.Adam(reseau_0.parameters(),lr=10e-6)
criterion = torch.nn.BCEWithLogitsLoss()
all_losses = []
all_correct = []
EPOCH = 5
for epoch in range(EPOCH):
    for i, (data, label) in enumerate(train_loader):
        if(i%100 == 0):
            print("J'ai fait",i,"étapes")
        pred = reseau_0(data)
        loss = criterion(pred,label)
        all_losses.append(loss)
        loss.backward()
        optimizer.step()
        #print(pred.shape)
        pred = pred.transpose(0,1)
        new_pred = torch.tensor([torch.argmax(pred_ind) for pred_ind in pred])
        correct = (label.eq(new_pred.long())).sum()
        all_correct.append(correct)

plt.plot(all_losses)
plt.show()

plt.plot(all_correct)
plt.show()
"""
ds = prepare_dataset("edu.standford.aclimdb")
word2id, embeddings = prepare_dataset('edu.standford.glove.6b.%d' % EMBEDDING_SIZE).load()

class FolderText(Dataset):
    def __init__(self, classes, tokenizer, load=False):
        self.tokenizer = tokenizer
        self.files = []
        self.filelabels = []
        self.labels = list(classes.keys())
        for label, folder in classes.items():
            for file in folder.glob("*.txt"):
                self.files.append(file)
                self.filelabels.append(label)

    def __len__(self):
        return len(self.filelabels)
    
    def __getitem__(self, ix):
        return self.tokenizer(self.files[ix].read_text()), self.filelabels[ix]


WORDS = re.compile(r"\S+")
def tokenizer(t):
    return list([x for x in re.findall(WORDS, t.lower())])

train_data = FolderText(ds.train.classes, tokenizer, load=False)
test_data = FolderText(ds.test.classes, tokenizer, load=False)
"""