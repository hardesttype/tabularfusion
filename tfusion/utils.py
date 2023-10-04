import pandas as pd
import numpy as np
from typing import List, Union

def datetime_series_clip(
    series: pd.Series,
    lower_bound: Union[np.datetime64, str] = np.datetime64('1677-09-21', 'D'),
    upper_bound: Union[np.datetime64, str] = np.datetime64('2262-04-11', 'D'),
    copy: bool = True,
) -> pd.Series:
    """Perform datetime clipping on a pandas Series.

    This function clips (limits) the datetime values within a specified lower
    and upper bound while handling various forms of missing or invalid data
    gracefully.

    The default bounds are chosen to be fit `'datetime64[ns]'` format,
    but you can customize them as needed.

    Parameters:
    - series (pd.Series): The pandas Series containing datetime values to be clipped.
    - lower_bound (Union[np.datetime64, str], optional):
      The lower bound to which datetime values should be clipped.
      Default is '1677-09-21'. Any datetime values
      before this date will be set to this lower bound.
    - upper_bound (Union[np.datetime64, str], optional):
      The upper bound to which datetime values should be clipped.
      Default is '2262-04-11'. Any datetime values
      after this date will be set to this upper bound.
    - copy (bool, optional): If True, it creates a copy of the input Series.
      If False, the operation is performed in without copying (faster).
      Default is True.

    Returns:
    - pd.Series: A new pandas Series with the same index and name as the input
      series but with datetime values clipped within the specified bounds. The
      dtype of the returned Series is set to 'datetime64[ns]'.

    Example:
    ```python
    import pandas as pd
    import numpy as np

    # Create a pandas Series with datetime values
    date_series = pd.Series(['2023-01-01', pd.NaT, '9999-01-01'])

    print(date_series)

    # Mixed dtype:
    0   2023-01-01
    1          NaT
    2   9999-01-01
    dtype: object

    # Clip the datetime values within a specific range
    clipped_series = datetime_series_clip(date_series)

    print(clipped_series)

    # Output:
    0   2023-01-01
    1          NaT
    2   2262-04-11
    dtype: datetime64[ns]
    ```
    """
    arr = series.astype(str, copy=copy).fillna('nat')\
    .replace({
        'None': 'nat', '': 'nat', 'NaN': 'nat',
        'null': 'nat', 'nan': 'nat', 'NULL': 'nat'
    })
    arr = np.array(arr, dtype = 'datetime64[D]')

    mask = ~np.isnat(arr)
    arr[mask] = np.clip(
        arr[mask],
        lower_bound,
        upper_bound,
    )
    return pd.Series(arr, index=series.index, name=series.name, dtype='datetime64[ns]')


def truncate_repeated_characters(
        tokens: List[int],
        normalize: bool = False,
        ):
        """
        Truncates sequence with repeated characters
        Args:
            input: list of torch.tensor
            normalize: normalize counts by input lengths

        Example:
            >>> tokens_list = [[1, 5, 5, 5, 4, 5, 5, 7, 7, 2], [1, 4, 4, 2]]
            >>> output_chars, output_counts = truncate_repeated_characters(tokens_list)
            >>> output_chars
            [[1, 5, 4, 5, 7, 2], [1, 4, 2]]
            >>> output_counts
            [[1, 3, 1, 2, 2, 1], [1, 2, 1]]
        """
        output_chars = []
        output_counts = []

        for tensor in tokens:
            chars = []
            counts = []
            count = 1
            prev_char = tensor[0]

            for char in tensor[1:]:
                if char == prev_char:
                    count += 1
                else:
                    chars.append(prev_char)
                    counts.append(count)
                    prev_char = char
                    count = 1

            chars.append(prev_char)
            counts.append(count)

            output_chars.append(chars)
            output_counts.append(counts)

        if normalize:
            output_counts = [out / len(seq) for out, seq in zip(output_counts, tokens)]

        return output_chars, output_counts
