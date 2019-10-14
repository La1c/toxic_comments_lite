from sklearn.base import BaseEstimator, TransformerMixin

class BaseFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        raise NotImplementedError

    def transform(self, X, y=None):
        raise NotImplementedError

    def fit_transform(self, X, y=None):
        raise NotImplementedError

    def set_params(self, **params):
        raise NotImplementedError

    def get_params(self, deep=True):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    @classmethod
    def load(cls, path):
        raise NotImplementedError