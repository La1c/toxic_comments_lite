from sklearn.naive_bayes import MultinomialNB
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator, TransformerMixin
import pickle
from featurizers.base_featurizer import BaseFeaturizer
import os

class MNBFeaturizer(BaseFeaturizer):
    def __init__(self, tfidf_vectorizer):
        self.tfidf = tfidf_vectorizer
        self.mnb = MultinomialNB()
        self.is_fitted = False
        super().__init__()

    def fit(self, X, y=None):
        self.mnb.fit(self.tfidf.transform(X), y)
        self.is_fitted = True
        return self

    def transform(self, X, y=None):
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
        with open(os.path.join(path, 'mnb.pkl'),'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        instance = None
        with open(os.path.join(path, 'mnb.pkl'), 'rb') as f:
            instance = pickle.load(f)
        return instance