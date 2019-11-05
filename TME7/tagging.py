import itertools
import logging
from tqdm import tqdm
import unicodedata
import string

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch


### Partie Tagging

logging.basicConfig(level=logging.INFO)

from datamaestro import prepare_dataset

ds = prepare_dataset('org.universaldependencies.french.gsd')

BATCH_SIZE = 100


# Format de sortie
# https://pypi.org/project/conllu/

class VocabularyTagging:
    OOVID = 0
    
    def __init__(self, oov: bool):
        self.oov = oov
        if oov:
            self.word2id = {"__OOV__": Vocabulary.OOVID}
            self.id2word = ["__OOV__"]
        else:
            self.word2id = {}
            self.id2word = []
    
    def __getitem__(self, i):
        return self.id2word[i]
    
    def get(self, word: str, adding=True):
        try:
            return self.word2id[word]
        except KeyError:
            if adding:
                wordid = len(self.id2word)
                self.word2id[word] = wordid
                self.id2word.append(word)
                return wordid
            if self.oov:
                return Vocabulary.OOVID
            raise
    
    def __len__(self):
        return len(self.id2word)


class TaggingDataset():
    def __init__(self, data, words: VocabularyTagging, tags: VocabularyTagging, adding=True):
        self.sentences = []
        
        for s in data:
            self.sentences.append(([words.get(token["form"], adding) for token in s],
                                   [tags.get(token["upostag"], adding) for token in s]))
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, ix):
        return self.sentences[ix]


logging.info("Loading datasets...")
words = VocabularyTagging(True)
tags = VocabularyTagging(False)
train_data = TaggingDataset(ds.files["train"], words, tags, True)
dev_data = TaggingDataset(ds.files["dev"], words, tags, True)
test_data = TaggingDataset(ds.files["test"], words, tags, False)

logging.info("Vocabulary size: %d", len(words))
