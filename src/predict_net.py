import torch
import argparse
from bert_classifier import BertSentimentClassifier
from bert_dataloader import BertDataset
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import os

def predict(model_path, test_data_path, maxlen, batch_size):
    bert = BertSentimentClassifier()
    bert.load_state_dict(torch.load(model_path))
    test_dataset = BertDataset(test_data_path, maxlen, is_training=False)
    data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers = 0)
    
    test_df = pd.read_csv(test_data_path)
    df = pd.DataFrame([], columns=["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])
    
    
    with torch.no_grad():
        for seq, attn_masks, labels in tqdm(data_loader):
            pred = pd.DataFrame(torch.sigmoid(bert(seq, attn_masks)).numpy(), columns=["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])
            df = pd.concat([df,pred], ignore_index=True)

    df['id'] = test_df['id']
    cols = df.columns.to_list()[-1:] + df.columns.to_list()[:-1]
    df = df[cols]
    return df

    
    
if __name__ == "__main__":
    from utils import try_mkdir
    
    parser = argparse.ArgumentParser()
    parser.add_argument('test_file', type=str)
    parser.add_argument('model_file',type=str)
    parser.add_argument('prediction_file', type=str)
    parser.add_argument('--maxlen', default=30, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    args = parser.parse_args()  
    
    df = predict(args.model_file,
            args.test_file,
            args.maxlen,
            args.batch_size)
    
    print(df.head())
    df.to_csv(args.prediction_file, index=False)