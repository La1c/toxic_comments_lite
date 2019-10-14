from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('input_folder_path', type=str)
args = parser.parse_args()  

column_name_map = {1: "text", 2: "label_toxic", 3: "label_severe_toxic", 4:"label_obscene", 5:"label_threat", 6:"label_insult", 7:"label_identity_hate"}
corpus = CSVClassificationCorpus(args.input_folder_path,
                                 column_name_map,
                                 skip_header=True,
                                 delimiter=',')
