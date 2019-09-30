from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator, TransformerMixin
import pickle
import os
import pandas as pd
from utils import try_mkdir

class MNBFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, tfidf_vectorizer):
        self.tfidf = tfidf_vectorizer
        self.mnb = MultinomialNB()
        self.is_fitted = False
        super().__init__()

    def fit(self, X, y):
        self.mnb.fit(self.tfidf.transform(X), y)
        self.is_fitted = True
        return self

    def transform(self, X):
        if not self.is_fitted:
            raise NotFittedError('Featurizer is not fitted yet')

        tfidf_matrix = self.tfidf.transform(X)
        feature_probs = self.mnb.feature_log_prob_[0]
        feature_matrix = tfidf_matrix.multiply(feature_probs)
        return feature_matrix

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def save(self, path):
        with open(os.path.join(path, 'tfidf.pkl'),'wb') as f:
            pickle.dump(self.tfidf, f, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(path, 'mnb.pkl'),'wb') as f:
            pickle.dump(self.mnb, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        tf_idf = None
        mnb = None

        with open(os.path.join(path, 'tfidf.pkl'), 'rb') as f:
            tf_idf = pickle.load(f)

        with open(os.path.join(path, 'mnb.pkl'), 'rb') as f:
            mnb = pickle.load(f)

        instance = MNBFeaturizer(tf_idf)
        instance.mnb = mnb

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
        data_df = pd.read_csv(args.input_data_path)
        try_mkdir(args.output_data_path)
        for category in ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]:
            featurizer = MNBFeaturizer.load(os.path.join(args.artifacts_path, category))
            transformed = featurizer.transform(data_df['comment_text'])
            
            with open(os.path.join(args.output_data_path, '{}_features_test.pkl'.format(category)), 'wb') as f:
                pickle.dump(transformed, f)

    else:
        print('Reading data from {}'.format(args.input_data_path))
        data_df = pd.read_csv(args.input_data_path)
        tfidf = TfidfVectorizer()
        tfidf.fit(data_df['comment_text'])

        try_mkdir(args.artifacts_path)
        try_mkdir(args.output_data_path)

        for category in ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]:
            featurizer = MNBFeaturizer(tfidf)
            transformed = featurizer.fit_transform(data_df['comment_text'], data_df[category])
            with open(os.path.join(args.output_data_path, '{}_features_train.pkl'.format(category)), 'wb') as f:
                pickle.dump(transformed, f)
            
            try_mkdir(os.path.join(args.artifacts_path, category))
            featurizer.save(os.path.join(args.artifacts_path, category))
            


            


            


