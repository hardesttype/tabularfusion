import pandas as pd
import numpy as np
from typing import List


class NumProcessor(DataProcessor):
    """Process numeric variables.

    This class is designed to process numeric variables in a pandas DataFrame. It provides various options
    for data preprocessing, including scaling, logging, clipping, and handling missing values.

    Parameters:
    - column_name (str): The name of the column containing the numeric data to be processed.
    - scaler (str, optional): The type of scaling to apply ('standard' or 'minmax'). Default is 'standard'.
    - log (bool, optional): If True, apply natural logarithm transformation to the data. Default is False.
    - clip (str, optional): The type of clipping to apply ('minmax' for min-max scaling range or None for no clipping). Default is None.
    - clip_min (float, optional): The minimum value for clipping. Default is None.
    - clip_max (float, optional): The maximum value for clipping. Default is None.
    - fillna (bool, optional): If True, fill missing values with a specified fill_value. Default is False.
    - fill_value (float, optional): The value to fill missing data with if fillna is True. Default is 0.0.
    - eps (float, optional): A small epsilon value to prevent division by zero. Default is 1e-6.
    - group_column (str, optional): The name of a column used for grouping data when calculating scalers. Default is None.
    - copy (bool, optional): If False, perform transformations in place, modifying the input DataFrame. Default is True.

    Methods:
    - fit(x): Fit the NumProcessor to the input DataFrame x. This calculates scaling parameters and other necessary values.
    - transform(x): Transform the input DataFrame x based on the previously calculated parameters.

    Attributes:
    - subtrahend (float): The value subtracted during scaling.
    - divider (float): The value used for division during scaling.

    Example:
    ```
    import pandas as pd
    from num_processor import NumProcessor

    data = pd.DataFrame({'Age': [25, 30, 35, None, 40],
                         'Income': [50000, 60000, 70000, 55000, None]})

    processor = NumProcessor(column_name='Age', scaler='minmax', log=True, fillna=True, fill_value=0.0)
    processed_data = processor.fit_transform(data)

    print(processed_data)

    # Output:
            Age   Income
    0  0.877348  50000.0
    1  0.924713  60000.0
    2  0.964979  70000.0
    3  0.000000  55000.0
    4  1.000000      NaN
    ```
    """
    def __init__(
        self,
        column_name:   str,
        scaler:        str   = 'standard',
        log:           bool  = False,
        clip:          str   = None,
        clip_min:      float = None,
        clip_max:      float = None,
        fillna:        bool  = False,
        fill_value:    float = 0.,
        eps:           float = 1e-6,
        group_column:  str   = None,
        copy:          bool  = True,
    ):
        self.column_name   = column_name
        self.scaler        = scaler
        self.log           = log
        self.clip          = clip
        self.fillna        = fillna
        self.eps           = eps
        self.group_column  = group_column
        self.copy          = copy

        self.fill_value = fill_value
        self.clip_min   = clip_min
        self.clip_max   = clip_max
        self.subtrahend = np.nan
        self.divider    = np.nan

    def __scale(self, data: pd.Series, func: str, group: pd.Series = None):
        if self.group_column is None:
            return data.agg(func)
        else:
            return data.groupby(group).agg(func).mean()

    def fit(self, x: pd.DataFrame):
        data = x.loc[:, self.column_name].astype(float, copy=False)

        if self.fillna:
            data = data.fillna(value=self.fill_value)

        if self.clip in ['minmax']:
            self.clip_min = data.min()
            self.clip_max = data.max()
            data = data.clip(lower=self.clip_min, upper=self.clip_max)

        elif self.clip_min is not None or\
             self.clip_max is not None:
            data = data.clip(lower=self.clip_min, upper=self.clip_max)

        if self.log:
            data = np.log1p(data.abs()) * np.sign(data)

        if self.scaler in ['standard']:
            self.subtrahend = self.__scale(data, 'mean', x.get(self.group_column))
            self.divider    = self.__scale(data, 'std',  x.get(self.group_column))

        elif self.scaler in ['minmax']:
            self.subtrahend = self.__scale(data, 'min', x.get(self.group_column))
            self.divider    = self.__scale(data, 'max', x.get(self.group_column)) -\
                              self.subtrahend
        else:
            self.subtrahend = 0
            self.divider = 1

        self.divider = self.divider if self.divider > self.eps else 1

        return self

    def transform(self, x: pd.DataFrame):
        data = x.loc[:, self.column_name].astype(float, copy=False)

        if self.fillna:
            data = data.fillna(value=self.fill_value)

        if self.clip in ['minmax']:
            self.clip_min = data.min()
            self.clip_max = data.max()
            data = data.clip(lower=self.clip_min, upper=self.clip_max)

        elif self.clip_min is not None or\
             self.clip_max is not None:
            data = data.clip(lower=self.clip_min, upper=self.clip_max)

        if self.log:
            data = np.log1p(data.abs()) * np.sign(data)

        data = (data - self.subtrahend) / self.divider

        if not self.copy:
            x.loc[:, self.column_name] = data
            return x
        else:
            return x.assign(**{str(self.column_name): data})


class CatProcessor(DataProcessor):
    """
    Process categorical variables.

    This class is designed to process categorical variables in a pandas DataFrame. It provides two options for
    encoding categorical data: identifying unique categories or frequency encoding.

    Parameters:
    - column_name (str): The name of the column containing the categorical data to be processed.
    - encoding (str, optional): The encoding method to use ('identity' or 'frequency'). Default is 'frequency'.
    - fillna (bool, optional): If True, fill missing values with a specified fill_value. Default is False.
    - fill_value (str or int, optional): The value to fill missing data with if fillna is True. Default is None.
    - copy (bool, optional): If False, perform transformations in place, modifying the input DataFrame. Default is True.

    Methods:
    - fit(x): Fit the CatProcessor to the input DataFrame x. This identifies unique categories or calculates
      frequency encodings based on the input data.
    - transform(x): Transform the input DataFrame x based on the previously calculated encodings.

    Example:
    ```python
    import pandas as pd
    from cat_processor import CatProcessor

    data = pd.DataFrame({'Category': ['A', 'B', 'A', 'C', 'B', None]})

    processor = CatProcessor(column_name='Category', encoding='frequency', fillna=True, fill_value='Unknown')
    processor.fit(data)
    processed_data = processor.transform(data)
    ```

    Notes:
    - The `fit` method identifies unique categories or calculates frequency encodings based on the input data and specified options.
    - The `transform` method applies the specified encoding to the input DataFrame.
    - The `unique_categories` attribute stores the unique categories identified during fitting, which can be useful for inspection or external use.
    - The `encoding_map` attribute stores the mapping of categories to their frequency encodings.
    """

    def __init__(
        self,
        column_name: str,
        encoding: str = 'frequency',
        n_cat: int = None,
        fillna: bool = False,
        fill_value = None,
        copy: bool = True,
    ):
        self.column_name = column_name
        self.encoding = encoding
        self.n_cat = n_cat
        self.fillna = fillna
        self.fill_value = fill_value
        self.copy = copy

        self.encoding_map = {}
        self.other = 1

    def fit_dtype(self):
        if self.other <= np.iinfo(np.int8).max:
            self.dtype = np.int8
        elif self.other <= np.iinfo(np.int16).max:
            self.dtype = np.int16
        elif self.other <= np.iinfo(np.int32).max:
            self.dtype = np.int32
        else:
            self.dtype = np.int64

    def fit(self, x: pd.DataFrame):
        data = x.loc[:, self.column_name]

        if self.fillna:
            data.fillna(value=self.fill_value, inplace=True)

        # 0 - padding
        if self.encoding == 'frequency':
            categories = data.value_counts(dropna=False)[:self.n_cat]
            categories = pd.Series(
                np.arange(categories.shape[0]) + 1,
                index=categories.index
            )
            self.encoding_map = categories.to_dict()
            self.other = categories.shape[0] + 1
            self.n_cat = categories.shape[0]

        self.fit_dtype()

        return self

    def transform(self, x: pd.DataFrame):
        data = x.loc[:, self.column_name]

        if self.fillna:
            data.fillna(value=self.fill_value, inplace=True)

        if self.encoding == 'frequency':
            data = data.map(self.encoding_map).fillna(self.other).astype(self.dtype, copy=False)

        if not self.copy:
            x.loc[:, self.column_name] = data
            return x
        else:
            return x.assign(**{str(self.column_name): data})
