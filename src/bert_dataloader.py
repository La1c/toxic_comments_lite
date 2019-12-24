import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd
import numpy as np

class BertDataset(Dataset):
    def __init__(self, filename, maxlen, is_training = True, text_cloumn= 'comment_text', label_columns=['toxic','severe_toxic','obscene','threat','insult','identity_hate'], bert_model= 'bert-base-uncased'):
        self.df = pd.read_csv(filename)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.maxlen = maxlen
        self.comment_column = text_cloumn
        self.label_columns = label_columns
        self.is_training = is_training
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        sentence = self.df.loc[index, self.comment_column]
        labels = None
        if self.is_training:
            labels = np.array(self.df.loc[index, self.label_columns].values, dtype=int)
        else:
            labels = np.array([0]*len(self.label_columns), dtype=int)
        
        tokens = self.tokenizer.tokenize(sentence)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        
        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))]
        else:
            tokens = tokens[:self.maxlen-1] + ['[SEP]']
            
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_ids_tensor = torch.tensor(tokens_ids)
        attn_mask = (tokens_ids_tensor != 0).long()
        return tokens_ids_tensor, attn_mask, labels
        
        