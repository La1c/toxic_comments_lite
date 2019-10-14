from flair.data import Corpus, Sentence
from flair.datasets import CSVClassificationCorpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('model_path', type=str)
args = parser.parse_args()  


classifier = TextClassifier.load(args.model_path)

# create example sentence
sentence = Sentence('France is the current world cup winner.')

# predict class and print
classifier.predict(sentence)

print(sentence.labels)