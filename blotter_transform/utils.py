import math
from collections import Counter

import numpy as np
import pandas as pd
from functools import wraps
from fuzzywuzzy import fuzz, process


class StringUtils:
    """
    A utility class for string operations.

    Methods
    -------
    strip_lower(val: str) -> str:
        Returns a lowercased string with leading/trailing whitespace removed.
    """

    @staticmethod
    def strip_lower(val: str):
        """
        Return a lowercased string with leading/trailing whitespace removed.

        Parameters
        ----------
        val : str
            The string to be stripped and lowercased.

        Returns
        -------
        str
            The stripped and lowercased string.
        """
        return val.strip().lower()

    @staticmethod
    def find_string_indices(series):
        """
        Given a Pandas BlotterSeries, finds the indices which are
        of type str

        Parameters:
        -----------
        int_vals : pandas.BlotterSeries
            A BlotterSeries containing integer values.

        Returns:
        --------
        list
            A list of indices of type String
        """
        return series[
            series.apply(lambda x: isinstance(x, int)) == False
        ].index.to_list()


class MathUtils:
    """
    A utility class for mathematical operations.

    Methods
    -------
    compute_z_score(dates)
        Computes the z-score for the given dates.
    """

    @staticmethod
    def compute_z_score(dates):
        """
        Computes the z-score for the given dates.
        (val - mean of population) / standard deviation

        Parameters
        ----------
        dates : pandas.BlotterSeries
            A  pandas BlotterSeries containing the dates.

        Returns
        -------
        numpy.ndarray
            A pandas BlotterSeries containing the z-scores for the given dates.
        """
        return np.abs((dates - dates.mean()) / dates.std())

    @staticmethod
    def calculate_return(row, bod_capital):
        """ """
        return row['pal'] / \
            bod_capital.loc[bod_capital['date'] == row['date'], 'exposure'].iloc[0]


class DateUtils:
    """
    A collection of utility functions for working with dates and times.

    Methods:
    --------
    cast_int(val)
        Attempts to cast a value to an integer. If the value cannot be
        cast to an integer, returns a pandas NaT value instead.

    get_outliers(dates, z_score_threshold=4)
        Given a Pandas BlotterSeries of dates, returns a list of dates that are
        considered outliers based on a z-score threshold.

    replace_with_neighbor(outlier_idx, dates, outliers: list, upper=True)
        Given an outlier index, a Pandas BlotterSeries of dates, and a list of
        outlier indices, replaces the outlier date with a neighboring
        date. If both neighboring dates are outliers, the original date
        is passed. Returns a tuple with the index of the replaced date
        and the new date value.

    cleanup_dates(date)
        Given a Pandas BlotterSeries of dates, performs data cleaning and
        returns a new BlotterSeries of dates with all values cast as integers.
    """

    @staticmethod
    def cast_int(val):
        """
        Cast a value to an integer.

        Parameters:
        -----------
        val : int, float, str, or other value
            The value to be cast to an integer.

        Returns:
        --------
        int or NaT
            If the value can be cast to an integer, returns the integer
            value. If the value cannot be cast to an integer, returns a
            pandas NaT value instead.
        """
        try:
            return int(val)
        except Exception:
            return pd.NaT

    @staticmethod
    def get_outliers(dates, z_score_threshold=4) -> list:
        """
        Given a Pandas BlotterSeries of dates, returns a list of dates that are
        considered outliers based on a z-score threshold.

        Parameters:
        -----------
        dates : pandas.BlotterSeries
            A Pandas BlotterSeries of dates to check for outliers.
        z_score_threshold : int or float, optional
            The threshold for the z-score above which a date is considered
            an outlier. Defaults to 4.

        Returns:
        --------
        list
            A list of dates that are considered outliers based on the
            z-score threshold.
        """
        zscores = MathUtils.compute_z_score(dates)
        # filter for values bigger than zscore mean * threshold val
        return dates[zscores > z_score_threshold * zscores.mean()]

    @staticmethod
    def replace_with_neighbor(outlier_idx, dates, outliers: list, upper=True):
        """
        Given an outlier index, a Pandas BlotterSeries of dates, and a list of
        outlier indices, replaces the outlier date with a neighboring
        date. If both neighboring dates are outliers, the original date
        is passed. Returns a tuple with the index of the replaced date
        and the new date value.

        Parameters:
        -----------
        outlier_idx : int
            The index of the outlier date to be replaced.
        dates : pandas.BlotterSeries
            A Pandas BlotterSeries of dates.
        outliers : list
            A list of indices of outlier dates.
        upper : bool, optional
            If True, replaces the outlier date with the upper neighboring
            date. If False, replaces the outlier date with the lower
            neighboring date. Defaults to True.

        Returns:
        --------
        tuple
            A tuple with the index of the replaced date and the new date
            value.
        """
        if outlier_idx == 0:
            # we can only take the next value if its not in the outliers
            if outlier_idx + 1 not in outliers:
                return (outlier_idx, dates[outlier_idx + 1])
        elif outlier_idx == len(dates) - 1:
            # we can only take the previous value
            if outlier_idx - 1 not in outliers:
                return (outlier_idx, dates[outlier_idx - 1])
        else:
            # we find the middle value
            if outlier_idx + 1 not in outliers and outlier_idx - 1 not in outliers:
                mean_val = np.mean([(dates[outlier_idx - 1], dates[outlier_idx + 1])])
                date_val = math.ceil(mean_val) if upper else math.floor(mean_val)
                return (outlier_idx, date_val)
            # handling multiple consecutive outliers
            forward_ptr = outlier_idx
            while forward_ptr in outliers:
                forward_ptr += 1
            mean_val = np.mean([(dates[outlier_idx - 1], dates[forward_ptr])])
            date_val = math.ceil(mean_val) if upper else math.floor(mean_val)
            return (outlier_idx, date_val)
        return outlier_idx, False

    @staticmethod
    def set_idx_to_nan(date, indx):
        """
        Set indices of pandas BlotterSeries date, to np.NaN

        Parameters:
        -----------
        date : pandas.BlotterSeries
            A BlotterSeries containing date values.
        indx : list
            A list of indices.

        Returns:
        --------
        pandas.BlotterSeries
            The cleaned BlotterSeries of dates.
        """
        if indx != []:
            date[indx] = np.NaN
            date = date.astype(float)
        return date

    @staticmethod
    def find_malformed_indices(date) -> list:
        """
        Given a Pandas BlotterSeries of dates, performs data cleaning and
        returns a new BlotterSeries of dates with all values cast as integers.

        Parameters:
        -----------
        dates : pandas.BlotterSeries
            A BlotterSeries containing date values.

        Returns:
        --------
        pandas.BlotterSeries
            A new BlotterSeries with updated date values.
        """
        # if a string is contained
        date_vals = date.apply(lambda x: DateUtils.cast_int(x))
        str_indx: list = StringUtils.find_string_indices(date_vals)
        date = DateUtils.set_idx_to_nan(date, str_indx)
        # find indices of NaN entries
        null_idx: list = date[date.isna() == True].index.to_list()
        # find outlier-idx from dates using z-score
        outliers = DateUtils.get_outliers(date.dropna())
        outliers_idx: list = outliers.index.to_list()
        # indices of NaN values, z-score outliers and strings
        return sorted(null_idx + outliers_idx + str_indx)

    @staticmethod
    def cleanup_dates(date, malformed_indices: list):
        """
        Given a pandas BlotterSeries date and corresponding malformed_indices,
        replaces the value at index with a neighboring value or mean.

        Parameters:
        -----------
        date : pandas.BlotterSeries
            A BlotterSeries containing date values.
        malformed_indices : list
            A list of indices to be updated.

        Returns:
            pandas.BlotterSeries
                The updated BlotterSeries of dates.
        """
        date = date.apply(lambda x: DateUtils.cast_int(x))
        for change_val in malformed_indices:
            update_idx, update_val = DateUtils.replace_with_neighbor(
                outlier_idx=change_val, dates=date, outliers=malformed_indices
            )
            date[update_idx] = update_val
            print(f"update idx {update_idx} in column {'date'}, to: {update_val}")
        # cast to int
        return date.astype(int)


class TextUtils:
    """
    A class providing text-related utility functions.

    Methods
    -------
    compute_likely_entries(data, similarity_threshold=80):
        Computes likely entries for a dataset by replacing similar entries with their most common variant.
        Parameters:
            data (pandas.BlotterSeries): The dataset to process.
            similarity_threshold (int, optional): The similarity threshold (in percentage) above which two entries
                are considered similar. Defaults to 80.
        Returns:
            pandas.BlotterSeries or list: The processed dataset.

    find_best_match(string, choices):
        Finds the best match for a string in a list of choices using fuzzy string matching.
        Parameters:
            string (str): The string to match.
            choices (list): The list of choices to match against.
        Returns:
            tuple: A tuple containing the best match and its matching score.
    """

    @staticmethod
    def compute_likely_entries(data, similarity_threshold=80):
        """
        Computes likely entries for a dataset by replacing similar entries with their most common variant.

        Parameters:
        -----------
        data : pandas.BlotterSeries or list
            The dataset to process.
        similarity_threshold : int, optional
            The similarity threshold (in percentage) above which two entries are considered similar.
            Defaults to 80.

        Returns:
        --------
        pandas.BlotterSeries or list
            The processed dataset.
        """
        # Count the instances of each entry
        counts = dict(Counter(data))
        # Calculate similarity score for each pair of keys
        keys = list(counts.keys())
        for i, key1 in enumerate(keys):
            for key2 in keys[i + 1 :]:
                ratio = fuzz.token_set_ratio(key1, key2)
                # If the similarity score is high enough
                if ratio > similarity_threshold:
                    # Replace the key with lower count with the higher count
                    if counts[key1] > counts[key2]:
                        data = data.replace(key2, key1)
                    else:
                        data = data.replace(key1, key2)
        return data

    @staticmethod
    def find_best_match(string, choices):
        """
        Finds the best match for a string in a list of choices using fuzzy string matching.

        Parameters:
        -----------
        string : str
            The string to match.
        choices : list
            The list of choices to match against.

        Returns:
        --------
        tuple
            A tuple containing the best match and its matching score.
        """
        return process.extract(string, choices, scorer=fuzz.token_sort_ratio)


class DecoratorUtils:

    """
    TODO: duplicate code
    """

    @staticmethod
    def check_arg_notnull(class_arg):
        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                attr = getattr(self, class_arg, None)
                if attr is None or attr.empty:
                    raise ValueError(f"{class_arg} attribute is None")
                return func(self, *args, **kwargs)
            return wrapper
        return decorator

    @staticmethod
    def check_col_exists(col_name):
        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                if not hasattr(self, col_name):
                    raise ValueError(f"{col_name} column does not exist in DataFrame")
                return func(self, *args, **kwargs)
            return wrapper
        return decorator
