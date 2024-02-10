from typing import List, Dict
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .source import DataSource


class DataMart(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        data_sources: List[DataSource],
        copy: bool = True,
    ):
        self.data_sources = data_sources
        self.copy = copy

    def fit(self, data: Dict[str, pd.DataFrame]):
        for data_source in self.data_sources:
            data_source.fit(data[data_source.source_name])
        return self

    def transform(self, data: Dict[str, pd.DataFrame]):
        output = dict()
        for data_source in self.data_sources:
            data_source.copy = self.copy
            output[data_source.source_name] = data_source\
            .transform(data[data_source.source_name])

            if not self.copy and data_source.source_type == 'sequential':
                output[data_source.source_name]\
                .sort_values([
                    data_source.id_column, data_source.sort_column
                ], inplace=True)
            elif self.copy and data_source.source_type == 'sequential':
                output[data_source.source_name] =\
                output[data_source.source_name]\
                .sort_values([
                    data_source.id_column, data_source.sort_column
                ])

        return output

    def wrap(self, data: Dict[str, pd.DataFrame]):
        for data_source in self.data_sources:
            data_source.wrap(data[data_source.source_name])
        return self

    def __getitem__(self, index) -> pd.DataFrame:
        if type(index) == int:
            index = [index]

        output = dict()
        output[self.data_sources[0].id_column] = index
        for data_source in self.data_sources:
            output[data_source.source_name] = data_source[index]\
            .groupby(data_source.id_column, dropna=False, sort=False)\
            .apply(lambda x: self.construct_dict(x))\
            .reindex(index).apply(lambda x:
                pd.Series({
                    col: row if isinstance(row, np.ndarray)
                    else np.array([], dtype=data_source.dtypes[col])
                    for col, row in x.items()
                }), axis=1)[data_source.columns[2:]]\
            .to_dict(orient='list')
            output[data_source.source_name]['length'] =\
            pd.Series([data_source.length.get(i, 0) for i in index], index=index).values
        return output

    def convert_to_numpy_sequences(self, data: Dict[str, pd.DataFrame]):
        output = dict()
        for data_source in self.data_sources:
            data_source.copy = self.copy
            output[data_source.source_name] = data_source\
            .convert_to_numpy_sequences(data[data_source.source_name])

            output[data_source.source_name].columns =\
            pd.MultiIndex.from_product([
                [data_source.source_name], output[data_source.source_name].columns
            ], names=['source_name', 'column_name'])

        output = pd.concat(output.values(), axis=1)

        for data_source in self.data_sources:
            for col in output[data_source.source_name]:
                nan_values = output.loc[:, (data_source.source_name, col)].isna()
                if nan_values.sum() == 0:
                    continue

                if col == '_length':
                    output.loc[nan_values, (data_source.source_name, col)] = 0

                    max_value = output.loc[:, (data_source.source_name, col)].max()
                    new_dtype = data_source.select_int_dtype(max_value)

                    output.loc[:, (data_source.source_name, col)] =\
                    output.loc[:, (data_source.source_name, col)].astype(new_dtype, copy=False)

                else:
                    output.loc[nan_values, (data_source.source_name, col)] =\
                    output.loc[nan_values, (data_source.source_name, col)].apply(
                        lambda x: np.array([], dtype=data[data_source.source_name][col].dtype)
                    )

        output[('_meta', '_total_length')] = output.xs('_length', level='column_name', axis=1).sum(axis=1)

        return output

    @staticmethod
    def construct_dict(data):
        return pd.Series({
            k: v.values
            for k, v in data.to_dict(orient='series').items()
        })
