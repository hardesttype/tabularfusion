from typing import List
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from ..processor.base import DataProcessor


class DataSource(BaseEstimator, TransformerMixin):
    """
    Creates data source
    Args:
        source_name: Name of data source
        source_type: One of 'sequential' or 'flat'
        id_column: Column with identifier. For 'flat' data source
                   is unique for every row
        sort_columns: List of columns to sort values
                      for 'sequential' data source
        num_processors: List of data processors for numeric columns
        cat_processors: List of data processors for category columns
        txt_processors: List of data processors for text columns
    """
    __instance_count = 0

    def __init__(
        self,
        source_name:    str       = None,
        source_type:    str       = None,
        id_column:      str       = None,
        sort_column:    str       = None,
        num_processors: List[DataProcessor] = [],
        cat_processors: List[DataProcessor] = [],
        txt_processors: List[DataProcessor] = [],
        copy:           bool                = True,
    ):
        self.source_name    = self.__validate_source_name(source_name)
        self.source_type    = source_type.lower()
        self.id_column      = id_column
        self.sort_column    = sort_column
        self.num_processors = num_processors
        self.cat_processors = cat_processors
        self.txt_processors = txt_processors
        self.copy           = copy

        self.fitted: bool       = False
        self.data: pd.DataFrame = None
        self.index: pd.Series   = None

    def fit(self, X, y=None):
        for num_processor in self.num_processors:
            num_processor.fit(X)

        for cat_processor in self.cat_processors:
            cat_processor.fit(X)

        for txt_processor in self.txt_processors:
            txt_processor.fit(X)

        return self

    def transform(self, X):
        if not self.copy:
            import warnings
            warnings.filterwarnings('ignore')

        for num_processor in self.num_processors:
            num_processor.copy = self.copy
            X = num_processor.transform(X)

        for cat_processor in self.cat_processors:
            cat_processor.copy = self.copy
            X = cat_processor.transform(X)

        for txt_processor in self.txt_processors:
            cat_processor.copy = self.copy
            X = txt_processor.transform(X)

        if not self.copy:
            X.loc[:, X.columns] = X
            return X
        else:
            return X

    def __validate_source_name(self, source_name):
        DataSource.__instance_count += 1
        if source_name is None:
            return f"data_source_{DataSource.__instance_count}"
        else:
            return source_name

    def __new__(cls, *args, **kwargs):
        if   'source_type' in kwargs and kwargs['source_type'].lower() == 'sequential':
            return super(DataSource, cls).__new__(SequentialDataSource)
        elif 'source_type' in kwargs and kwargs['source_type'].lower() == 'flat':
            return super(DataSource, cls).__new__(FlatDataSource)
        else:
            raise ValueError('Invalid argument `source_type`')

    def column_list(self):
        columns = [self.id_column, self.sort_column]
        columns += [proc.column_name for proc in self.num_processors]
        columns += [proc.column_name for proc in self.cat_processors]
        columns += [proc.column_name for proc in self.txt_processors]

        return columns

    @staticmethod
    def select_int_dtype(max_value):
        if max_value <= np.iinfo(np.int8).max:
            dtype = np.int8
        elif max_value <= np.iinfo(np.int16).max:
            dtype = np.int16
        elif max_value <= np.iinfo(np.int32).max:
            dtype = np.int32
        else:
            dtype = np.int64
        return dtype

    def __getitem__(self, index) -> pd.DataFrame:
        if type(index) == int:
            index = [index]

        ids = pd.Series([
            self.inverted_index.get(i, np.nan)
            for i in index
        ], index=index).explode()

        return self.data.iloc[ids.dropna()]


class SequentialDataSource(DataSource):
    def wrap(self, X):
        self.data = X
        self.dtypes = self.data.dtypes
        self.index_dtype = self.select_int_dtype(len(self.data))
        self.inverted_index = self.data.loc[:, [self.id_column]]\
                    .assign(index=np.arange(len(self.data)))\
                    .groupby(self.id_column)['index'].apply(list)
        self.index = self.inverted_index.index
        self.length = self.inverted_index.apply(len).rename('length')
        self.length = self.length.astype(self.select_int_dtype(self.length.max()), copy=False)
        self.inverted_index = self.inverted_index\
                              .explode().astype(self.index_dtype, copy=False)
        self.columns = self.column_list()

        return self


class FlatDataSource(DataSource):
    def wrap(self, X):
        self.data = X
        self.dtypes = self.data.dtypes
        self.index_dtype = self.select_int_dtype(len(self.data))
        self.inverted_index = self.data.loc[:, [self.id_column]]\
                    .assign(index=np.arange(len(self.data)))\
                    .set_index(self.id_column)['index']\
                    .astype(self.index_dtype, copy=False)
        self.index = self.inverted_index.index
        self.length = pd.Series(
            np.ones(len(self.data)),
            index = self.inverted_index.index,
            name='length'
        )
        self.length = self.length.astype(self.select_int_dtype(self.length.max()), copy=False)
        self.columns = self.column_list()

        return self
