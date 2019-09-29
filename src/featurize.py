from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.exceptions import NotFittedError
import pickle
import os
import pandas as pd

class Featurizer:
    def __init__(self):
        self.tfidf = TfidfVectorizer()
        self.mnb = MultinomialNB()
        self.is_fitted = False

    def fit(self, X, y):
        self.tfidf.fit(X)
        self.mnb.fit(X, y)
        self.is_fitted = True


    def transform(self, X):
        if not self.is_fitted:
            raise NotFittedError('Featurizer is not fitted yet')

        tfidf_matrix = self.tfidf.transform(X)
        feature_probs = self.mnb.feature_log_prob_[0]
        feature_matrix = tfidf_matrix.multiply(feature_probs)
        return feature_matrix

    def save(self, path):
        with open(os.path.join(path, 'tfidf.pkl'),'w') as f:
            pickle.dump(self.tfidf, f, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(path, 'mnb.pkl'),'w') as f:
            pickle.dump(self.mnb, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        instance = Featurizer()

        with open(os.path.join(path, 'tfidf.pkl'), 'r') as f:
            instance.tfidf = pickle.load(f)

        with open(os.path.join(path, 'mnb.pkl'), 'r') as f:
            instance.mnb = pickle.load(f)

        instance.is_fitted = True

        return instance


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_data_path', type=str)
    parser.add_argument('output_data_path', type=str)
    parser.add_argument('artifacts_path', type=str)
    parser.add_argument('--train', help='Trains featurizer', action='store_true')
    parser.add_argument('--test', help='Transforms test data', action='store_true')
    args = parser.parse_args()

    if args.test:
        featurizer = Featurizer.load(args.artifacts_path)
        data_df = pd.read_csv(args.input_data_path)
        for category in ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]:
            transformed = featurizer.transform(data_df['comments_text'], data_df[category])
            with open(os.path.join(args.output_data_path, '{}_features_test.pkl'.format(category)), w) as f:
                pickle.dump(transformed, f)

    else:
        data_df = pd.read_csv(args.input_data_path)
        for category in ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]:
            featurizer = Featurizer()
            transformed = featurizer.fit(data_df['comments_text'], data_df[category])
            with open(os.path.join(args.output_data_path, '{}_features_test.pkl'.format(category)), w) as f:
                pickle.dump(transformed, f)


            


            


