import itertools
import logging
from tqdm import tqdm
import unicodedata
import string

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.nn import LSTM, CrossEntropyLoss
import numpy as np

#### Partie traduction

PAD = 0
EOS = 1
SOS = 2


class VocabularyTrad:
    def __init__(self):
        self.word2id = {"<PAD>": PAD, "<EOS>": EOS, "<SOS>": 2}
        self.id2word = {PAD: "<PAD>", EOS: "<EOS>", SOS: "<SOS>"}
    
    def get_sentence(self, sentence):
        return [self.get(x, True) for x in sentence.split(" ")] + [1]
    
    def get(self, w, adding=False):
        try:
            return self.word2id[w]
        except KeyError:
            if adding:
                self.word2id[w] = len(self.word2id)
                self.id2word[self.word2id[w]] = w
                return self.word2id[w]
            raise
    
    def __getitem__(self, i):
        return self.id2word[i]
    
    def __len__(self):
        return len(self.word2id)


def normalize(s):
    return ''.join(c if c in string.ascii_letters else " "
                   for c in unicodedata.normalize('NFD', s.lower().strip())
                   if c in string.ascii_letters + " " + string.punctuation)


class TradDataset(Dataset):
    def __init__(self, data, vocOrig, vocDest, adding=True, max_len=10):
        self.sentences = []
        for s in tqdm(data.split("\n")):
            if len(s) < 1: continue
            orig, dest = map(normalize, s.split("\t")[:2])
            if len(orig) > max_len: continue
            self.sentences.append((torch.tensor(vocOrig.get_sentence(orig)),
                                   torch.tensor(vocDest.get_sentence(orig))))
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, i):
        return self.sentences[i]


def my_coll(list_xy):
    list_x = [data[0] for data in list_xy]
    list_y = [data[1] for data in list_xy]
    
    argsort = sorted(range(len(list_xy)), key=lambda index: len(list_x[index]))
    list_x = [list_x[i] for i in argsort]
    list_y = [list_y[i] for i in argsort]
    
    print(list_x)
    print(list_y)
    list_x = pack_sequence(list_x)
    print(list_x)
    print(list_y)
    return list_x, list_y

def main():
    with open("fra.txt") as f:
        lines = f.read()
    
    vocEng = VocabularyTrad()
    vocFra = VocabularyTrad()
    datatrain = TradDataset(lines, vocEng, vocFra)
    for i in range(20000):
        if torch.any(datatrain[i][0] != datatrain[i][1]):
            print("diff")
    exit()
    
    
    size_in = datatrain[0][0].shape[0]
    size_hidden = 5
    num_layers = 3
    bi = 1 # 2 pour biLSTM
    batch_size = 8
    
    print(datatrain[0])
    
    dataloader = DataLoader(datatrain, collate_fn=my_coll, batch_size=batch_size)
    
    encoder = LSTM(input_size=size_in, hidden_size=size_hidden, num_layers=num_layers)
    decoder = LSTM(input_size=size_in, hidden_size=size_hidden, num_layers=num_layers)
    hidden_to_voc = nn.Linear(size_hidden, len(vocFra))
    
    optimEnc = Adam(encoder.parameters(), lr=1e-4)
    optimDec = Adam(decoder.parameters(), lr=1e-4)
    optimHidToVoc = Adam(hidden_to_voc.parameters(), lr=1e-4)
    optims = [optimEnc, optimDec, optimHidToVoc]
    
    criterion = CrossEntropyLoss()
    
    for x, y in dataloader:
        output, (h_n, c_n) = encoder(x)
        assert h_n.shape == (num_layers * bi, batch_size, size_hidden)
        assert c_n.shape == (num_layers * bi, batch_size, size_hidden)
        
        list_input = torch.stack((torch.tensor([SOS]), y))
        y_target = torch.stack((y, torch.tensor([EOS])))                                # FIXME
        _, (decoded_hidden, _) = decoder(input=list_input, h_0=h_n[-1], c_0=[-1])
        decoded_word = hidden_to_voc(decoded_hidden)
        
        for optim in optims:
            optim.zero_grad()
        
        loss = criterion(decoded_word, y_target)
        loss.backward()
        
        for optim in optims:
            optim.step()
        

if __name__ == '__main__':
    main()
