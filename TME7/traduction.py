from tqdm import tqdm
import unicodedata
import string
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as utilsRNN
import torch.nn as nn
from torch.optim import Adam
import torch.optim.lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import torch
from random import random
from time import time


PAD = 0
EOS = 1
SOS = 2


class VocabularyTrad:
    def __init__(self):
        self.word2id = {"<PAD>": PAD, "<EOS>": EOS, "<SOS>": SOS}
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
                                   torch.tensor(vocDest.get_sentence(dest))))
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, i):
        return self.sentences[i]


def pad_collate(batch):
    (xx, yy) = zip(*batch)
    argsort = sorted(range(len(batch)), key=lambda index: len(xx[index]), reverse=True)
    xx = [xx[i] for i in argsort]
    yy = [yy[i] for i in argsort]
    
    prefix = torch.tensor([SOS])
    
    xx = [torch.cat((prefix, x)) for x in xx]
    yy = [torch.cat((prefix, y)) for y in yy]
        
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]

    xx_pad = utilsRNN.pad_sequence(xx)
    yy_pad = utilsRNN.pad_sequence(yy)

    return (xx_pad, x_lens), (yy_pad, y_lens)


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_, lengths=None):
        """
        input:      [seq_len,  batch size]
        hidden:     [n_layers, batch size, hid_dim]
        cell:       [n_layers, batch size, hid_dim]"""
        
        embedded = self.dropout(self.embedding(input_))
        # embedded: [seq_len, batch size, emb_dim]
        
        if lengths is not None:
            embedded = utilsRNN.pack_padded_sequence(embedded, lengths)
        _outputs, (hidden, cell) = self.rnn(embedded)
        
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        
        self.out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_, hidden, cell, lengths=None):
        """
        input:      [seq_len,  batch size]
        hidden:     [n_layers, batch size, hid_dim]
        cell:       [n_layers, batch size, hid_dim]
        prediction: [seq_len,  batch size, output dim]"""
        
        embedded = self.dropout(self.embedding(input_))
        # embedded: [seq_len, batch size, emb_dim]
        
        if lengths is not None:
            embedded = utilsRNN.pack_padded_sequence(embedded, lengths, enforce_sorted=False)
        
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        if lengths is not None:
            output = utilsRNN.pad_packed_sequence(output)
        # output: [seq_len, batch size, hid_dim]
        
        prediction = self.out(output)
        
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert not encoder.rnn.bidirectional and not decoder.rnn.bidirectional,\
            "Le décodeur ne peut pas être bidirectionnel, donc l'encodeur non plus"
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """output: [max_len, batch, trg_voc_size]
        la première ligne ne contient que des zéros
        
        src: ([src sent len, batch size], len)
        trg: ([trg sent len, batch size], len)
        
        teacher_forcing_ratio is probability to use teacher forcing
        """
        if not self.training and teacher_forcing_ratio!=0:
            print("warning: teacher in eval mode")
        
        src, src_len = src
        trg, trg_len = trg
        assert src[0][0] == SOS
        assert trg[0][0] == SOS
        assert src[-1][0] == EOS
        assert torch.any(trg[-1, :] == EOS) # trg n'est pas trié
        
        batch_size = trg.shape[1]
        trg_max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_max_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src, src_len)
        
        #first input to the decoder is the <sos> tokens
        input_ = trg[0]
        
        for t in range(1, trg_max_len):
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            
            input_ = input_.unsqueeze(0)

            # lengths = (np.array(trg_len) < t) # 1 si la séquence n'est pas encore finie, 0 sinon
            # les séquences de taille 0 ne sont pas gérées
            output, hidden, cell = self.decoder(input_, hidden, cell)
            
            output = output.squeeze(0)
            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(dim=1)
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input_ = trg[t] if teacher_force else top1
        
        return outputs


def train(model, data, optimizer, criterion, clip, show=None):
    if show:
        src2word = show[0]
        trg2word = show[1]
    else:
        src2word = None
        trg2word = None
        
    model.train()
    
    epoch_loss = 0
    
    for i, ((src, src_len), (trg, trg_len)) in enumerate(data):
        optimizer.zero_grad()
        
        output = model((src, src_len), (trg, trg_len), teacher_forcing_ratio=1.)
        
        #trg = [trg sent len, batch size]
        #output = [trg sent len, batch size, output dim]
        
        if show and random()<.01:
            print([src2word[x.item()] for x in src[:, 0]])
            print([trg2word[x.item()] for x in torch.argmax(output[1:, 0], dim=1)])
            print([trg2word[x.item()] for x in trg[:, 0]])
            print()
        
        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)
        
        #trg = [(trg sent len - 1) * batch size]
        #output = [(trg sent len - 1) * batch size, output dim]
            
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(data)


def evaluate(model, data, criterion):
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
        for i, ((src, src_len), (trg, trg_len)) in enumerate(data):
            output = model((src, src_len), (trg, trg_len), teacher_forcing_ratio=0.)  #turn off teacher forcing
            #trg: [trg sent len, batch size]
            #output: [trg sent len, batch size, output dim]
            
            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)
            #trg: [(trg sent len - 1) * batch size]
            #output: [(trg sent len - 1) * batch size, output dim]
            
            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
    
    return epoch_loss / len(data)


def main():
    with open("fra.txt") as f:
        lines = f.read()
    
    SRC = VocabularyTrad()
    TRG = VocabularyTrad()
    dataset = TradDataset(lines, SRC, TRG)
    sizes = [int(len(dataset)*.7), len(dataset) - int(len(dataset)*.7)]
    dataset_train, dataset_test = torch.utils.data.random_split(dataset, sizes)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    INPUT_DIM = len(SRC)
    OUTPUT_DIM = len(TRG)
    batch_size = 8
    EMB_DIM = 50
    HID_DIM = 100
    N_LAYERS = 2
    ENC_DROPOUT = .5
    DEC_DROPOUT = .5
    
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, collate_fn=pad_collate)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, collate_fn=pad_collate)
    
    enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    
    
    model = Seq2Seq(enc, dec, device).to(device)
    optimizer = Adam(model.parameters())
    
    PAD_IDX = TRG.word2id['<PAD>']
    
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    N_EPOCHS = 100
    CLIP = 1
    writer = SummaryWriter()
    
    for epoch in range(N_EPOCHS):
        t0 = time()
        test_loss = evaluate(model, dataloader_test, criterion)
        
        train_loss = train(model, dataloader_train, optimizer, criterion, CLIP,
                           show=(SRC.id2word, TRG.id2word))
        
        writer.add_scalars('loss', {'train':train_loss, 'test':test_loss}, epoch)
        
        print("epoch", epoch, "/", N_EPOCHS, "%.1fs" % (time() - t0))
        
        lr_sched.step(epoch)

    # dataloader_perf = DataLoader(dataset, batch_size=len(dataset), collate_fn=pad_collate)
    # (x, x_len), _ = next(iter(dataloader_perf))
    # enc(x)
    # enc(x, x_len)
    # print(sum(x_len) / len(x_len))
    # print(x_len[:3], x_len[-3:])
    # print(x.shape)
    # 
    # t0 = time()
    # for _ in range(10):
    #     enc(x, x_len)
    # print(time() - t0)
    # t0 = time()
    # for _ in range(10):
    #     enc(x)
    # print(time() - t0)

if __name__ == '__main__':
    main()

