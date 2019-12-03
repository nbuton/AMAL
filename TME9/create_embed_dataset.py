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
import torch.nn as nn

def text_to_vector(texte,mot_vers_indice,train_target):
    all_review = texte.tolist()
    array_number = []
    print(len(all_review))
    print(train_target.shape)
    for num_review,review in enumerate(all_review):
        if(num_review%1000==0):
            print("j'ai traité",num_review,"phrases")
        review = review.replace("</br>","").replace("<br>","").lower()
        vecteurs = []
        for mot in review.split(" ") :
            try:
                indice = mot_vers_indice[mot]
            except:
                indice = 0
            vecteurs.append(indice)
        vecteurs = torch.tensor(vecteurs)
        vecteurs = (vecteurs, train_target[num_review])
        array_number.append(vecteurs)
    return array_number



glove_file = datapath('/home/nico/Documents/Cours M2 DAC/AMAL/TME9/glove.6B.50d.txt')
tmp_file = get_tmpfile("test_word2vec.txt")

_ = glove2word2vec(glove_file, tmp_file)
model = KeyedVectors.load_word2vec_format(tmp_file)
print("J'ai load le modele glove")
tmp_file = None


print(type(model.wv.vocab))
#Permet de faire le lien entre le mot et l'indice dans la matrice d'embedding
mot_vers_indice = {cle:valeur for valeur,cle in enumerate(model.vocab) }
print(mot_vers_indice["cat"])
print(model.wv["cat"])

#Matrice de transition de l'indice vers la représentation du mot
matrix = torch.tensor(model.wv.syn0)
print(matrix.shape)



df = pd.read_csv("IMDB Dataset.csv")
df = df.sample(20000)
print("J'ai load le dataset")
print(df.columns)
df["sentiment"] = pd.Categorical(df["sentiment"])
df['sentiment'] = df.sentiment.cat.codes

train_target = torch.tensor(df['sentiment'].values.astype(np.float32))
train = text_to_vector(df["review"],mot_vers_indice,train_target)
train_target = torch.tensor([])
print(len(train))
print(train_target.shape)

mot_vers_indice = None

with open('matrice_glove.pickle', 'wb') as handle:
    pickle.dump(matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

matrix = None
df = None
train = train[:20000]
with open('imdb_train.pickle', 'wb') as handle:
    pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)
