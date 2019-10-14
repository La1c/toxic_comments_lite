from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('input_folder_path', type=str)
parser.add_argument('output_model_path', type=str)
args = parser.parse_args()  

column_name_map = {1: "text", 2: "label_toxic"}#, 3: "label_severe_toxic", 4:"label_obscene", 5:"label_threat", 6:"label_insult", 7:"label_identity_hate"}
corpus = CSVClassificationCorpus(args.input_folder_path,
                                 column_name_map,
                                 train_file='train_prepared.csv',
                                 test_file='train_prepared.csv',
                                 skip_header=True,
                                 delimiter=',')

label_dict = corpus.make_label_dictionary()
#print(corpus.obtain_statistics())

word_embeddings = [WordEmbeddings('glove'),
                   #FlairEmbeddings('news-forward'),
                   #FlairEmbeddings('news-backward'),
                   ]
document_embeddings = DocumentRNNEmbeddings(word_embeddings,
                                            hidden_size=512,
                                            reproject_words=True,
                                            reproject_words_dimension=256,
                                            )

classifier = TextClassifier(document_embeddings,
                            label_dictionary=label_dict,
                            multi_label=True)

trainer = ModelTrainer(classifier, corpus)

trainer.train(args.output_model_path,
            #   learning_rate=0.1,
            #   mini_batch_size=32,
            #   anneal_factor=0.5,
            #   patience=5,
              max_epochs=1,
              num_workers=12)




