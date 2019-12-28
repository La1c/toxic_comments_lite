import argparse
from torch.utils.data.dataloader import DataLoader
from bert_classifier import BertSentimentClassifier
from bert_dataloader import BertDataset
import torch.nn as nn
import torch
import torch.optim as optim
import os
from utils import try_mkdir

def train_net(train_filename, freeze, maxlen, batch_size, max_epochs, n_jobs, prints_every):
    train_set = BertDataset(filename = train_filename, maxlen = maxlen)
    data_loader = DataLoader(train_set, batch_size = batch_size, num_workers = n_jobs)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = BertSentimentClassifier(freeze_bert = freeze)
    net.to(device)
    criterion = nn.BCEWithLogitsLoss()
    opt = optim.Adam(net.parameters(), lr = 2e-5)
    
    
    for ep in range(max_epochs):
        for it, (seq, attn_masks, labels) in enumerate(data_loader):
            opt.zero_grad()  
            seq, attn_masks, labels = seq.to(device), attn_masks.to(device), labels.to(device)

            logits = net(seq, attn_masks)

            loss = criterion(logits.squeeze(-1), labels.float())

            loss.backward()

            opt.step()

            if (it + 1) % prints_every == 0:
                print("Iteration {} of epoch {} complete. Loss : {}".format(it+1, ep+1, loss.item()))
                
    return net


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('train_filename', type=str)
    parser.add_argument('model_file',type=str)
    parser.add_argument('--freeze', default=True, type=bool)
    parser.add_argument('--maxlen', default=30, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--max_epochs', default=1, type=int)
    parser.add_argument('--n_jobs', default=0, type=int)
    parser.add_argument('--prints_every', default=10, type=int)
    
    
    args = parser.parse_args()  
    net = train_net(args.train_filename,
              args.freeze,
              args.maxlen,
              args.batch_size,
              args.max_epochs,
              args.n_jobs,
              args.prints_every)
    
    try_mkdir('/'.join(args.model_file.split('/')[:-1]))
    torch.save(net.state_dict(), args.model_file)









