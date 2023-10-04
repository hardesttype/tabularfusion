from sklearn.base import BaseEstimator, TransformerMixin


class DataProcessor(BaseEstimator, TransformerMixin):
    """
    Abstract class for data processors
    """
    def __init__(
        self,
        column_name: str,
    ):
        raise NotImplementedError('Abstract class!')
