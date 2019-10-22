import torch.nn as nn
from torch.optim import Adam
from tp5_preprocess import *
import gzip
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score

class Sentiment(nn.Module):
    def __init__(self, taille_embedding, taille_dico, conv_feature_maps, fc_layer_size):
        super(Sentiment, self).__init__()
        self.embed = nn.Embedding(taille_dico, taille_embedding)
        
        l = [nn.Conv1d(taille_embedding, conv_feature_maps[0], kernel_size=3, stride=1, padding=1)]
        for i in range(len(conv_feature_maps) - 1):
            l.append(nn.MaxPool1d(kernel_size=3, stride=2))
            l.append(nn.Conv1d(conv_feature_maps[i], conv_feature_maps[i+1], kernel_size=3, stride=1, padding=1))
        self.conv = nn.Sequential(*l)
        
        l = [nn.Linear(conv_feature_maps[-1], fc_layer_size[0])]
        for i in range(len(fc_layer_size) - 1):
            l.append(nn.Linear(fc_layer_size[i], fc_layer_size[i+1]))
        self.fc = nn.Sequential(*l)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        assert x.ndim == 2             # b, random
        b, r = x.shape
        
        x = self.embed(x)
        assert x.ndim == 3
        e = x.shape[2]
        assert x.shape == (b, r, e)    # b, random, tailleEmb 
        
        x = x.transpose(1,2)
        assert x.shape == (b, e, r)    # b, tailleEmb, random
        
        x = self.conv(x)
        f = x.shape[1]
        r = x.shape[2]
        assert x.shape == (b, f, r)    # b, feature_map_size, rand2
        
        x = torch.max(x,dim=2)[0]
        assert x.shape == (b, f)       # b, feature_map_size
        
        x = self.fc(x)
        assert x.shape == (b, 1)       # b, 1
        
        # x = self.sigmoid(x)
        # assert x.shape == (b, 1)       # b, 1
        
        x = x.squeeze(1)
        assert x.shape == (b, )        # b
        return x
    
    @staticmethod
    def score_to_class(x):
        return x > .5
    
def main_sentiment(taille_dico, train, test, n_epochs=1, i_max=-1):
    taille_embedding = 50
    myCNN = Sentiment(taille_embedding, taille_dico, conv_feature_maps=[10, 5], fc_layer_size=[5, 1])
    loss = nn.BCEWithLogitsLoss()
    optim = Adam(myCNN.parameters(), lr=1e-4)
    writer = SummaryWriter()
    
    test_data, test_target = test
    
    i = 0
    for epoch in range(n_epochs):
        print("epoch", epoch, "/", n_epochs)
        for data, target in train:
            if i % 100 == 0:
                print(i, "/", len(train))
                test_pred = myCNN(test_data)
                test_loss = loss(test_pred, test_target)
                writer.add_scalars('Loss_stochastique', {'test':test_loss.item()}, i)
                
                test_classes_pred = myCNN.score_to_class(test_pred)
                pre = accuracy_score(y_true=test_target, y_pred=test_classes_pred)
                writer.add_scalars('precision', {'test':pre}, i)
                
            if i == i_max:
                return 
            
            out = myCNN(data)
            l = loss(out, target.float())
            writer.add_scalars('Loss_stochastique', {'train':l.item()}, i)

            classes_pred = myCNN.score_to_class(out)
            pre = accuracy_score(y_true=target, y_pred=classes_pred)
            writer.add_scalars('precision', {'train': pre}, i)

            l.backward()
            optim.step()
            
            i+=1

def loaddata(f):
    with gzip.open(f,"rb") as fp:
        return torch.load(fp)

def main():
    taille_dico = 1000
    b_size = 32
    train = loaddata("train-1000.pth")
    print("taille train", len(train))
    train = DataLoader(train, batch_size=b_size, collate_fn=TextDataset.collate, shuffle=True)
    
    test = loaddata("test-1000.pth")
    test_size = min(100, len(test)) #todo rotating test
    test_x, test_y = next(iter(DataLoader(test, batch_size=test_size, collate_fn=TextDataset.collate, shuffle=True)))
    
    print("test data shape", test_x.shape)
    main_sentiment(taille_dico, train, (test_x, test_y.float()), i_max=10000)
    

if __name__ == '__main__':
    """
    en entrée texte (avec le sentiment comme label 1/0)
    On n'utilise pas de W2V mais des algorithmes 'subwords' pour gérer les nouveaux mots.
    Des choix possibles sont:
        - BPE : Byte Pair Encoding  ~=  WordPiece
        - Unigram Language Model : ?
    
    Ce qu'on fait:
        - BPE avec un vocabulaire de 1000 tokens pré-entraîné (librairie "SentencePiece")
        - padding avec des zéros pour que tous les éléments d'un batch aient la même taille
        - nn.Embedding(taille_dico=1000, taille_embedding=50):
            prend en entrée les indices des sous_mots
            apprend un embedding pour chaque sous-mot
        - Convolutions
        - Fully Connected
        - BCEWithLogits
        
    
    Dataset train annoté automatiquement mais test non donc pas de bonne perfs atteignables
    
    
    
    
    todo:
    w: kernel width
    s: stride
    l: longueur des entrées
    m: déplacement dans entrées tq dep de 1 dans sorties
    (l0, m0, w1, s1) -> (l1, m1)
    
    j: pos dans sortie
    -> pos dans entrée
    
    trouver ss-seq qui activent le plus chaque caractéristique de sortie
    """
    main()