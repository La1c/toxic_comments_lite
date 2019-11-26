import pandas as pd
from sklearn.metrics import roc_auc_score
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('predictions_path', type=str)
parser.add_argument('test_labels_path', type=str)
parser.add_argument('metric_file', type=str)
args = parser.parse_args()  

pred_df = pd.read_csv(args.predictions_path)
labels_df = pd.read_csv(args.test_labels_path)
labels_df = labels_df[labels_df.drop(columns=['id']).apply(lambda x: all(x.values >=0), axis = 1)]
    
merged_df = pd.merge(labels_df, pred_df, how='left', on='id',suffixes=('_true', '_pred'))

category_auc = []

for category in labels_df.drop(columns=['id']).columns:
    auc = roc_auc_score(merged_df[category + '_true'], merged_df[category + '_pred'])
    category_auc.append(auc)

mean_auc = np.mean(category_auc)

with open(args.metric_file, 'w') as f:
    f.write('{:4f}\n'.format(mean_auc))

