""" docstring sample """
__version__ = "0.1.0"

import warnings

import pandas as pd

from . import utils as u
from . import warning_messages as wm

# extending pandas
# https://pandas.pydata.org/pandas-docs/stable/development/extending.html

class Parameters:

    REQ_COL_NAMES = ["date", "lkid", "ticker", "analyst"]
    Z_SCORE_THRESHOLD = 3
    WARNING_STATUS = "ignore"

    @classmethod
    def set_warning_sttus(cls, new_param):
        warnings.filterwarnings(new_param)
        cls.WARNING_STATUS = new_param

    @classmethod
    def set_req_col_names(cls, new_param):
        cls.REQ_COL_NAMES = new_param

    @classmethod
    def set_z_score_threshold(cls, new_param):
        cls.Z_SCORE_THRESHOLD = new_param


class BlotterSeries(pd.Series):
    """
    Subclass of pandas.Series representing a single column of financial data,
    specialized for use in a trading blotter.

    Attributes
    ----------
    _constructor : property
        Constructor for creating new instances of this class.
    _constructor_expanddim : property
        Constructor for creating new instances of BlotterDataFrame.

    Methods
    -------
    normalize_dates() -> BlotterSeries:
        Returns a new instance of the BlotterSeries with normalized dates.
    normalize_texts() -> BlotterSeries:
        Returns a new instance of the BlotterSeries with normalized texts.
    """

    @property
    def _constructor(self):
        return BlotterSeries

    @property
    def _constructor_expanddim(self):
        return BlotterDataFrame

    def normalize_dates(self):
        malformed_indices = u.DateUtils.find_malformed_indices(self)
        return u.DateUtils.cleanup_dates(self, malformed_indices)

    def normalize_texts(self):
        return u.TextUtils.compute_likely_entries(self)


class BlotterDataFrame(pd.DataFrame):

    """
    A custom DataFrame class that extends the functionality of pandas DataFrame.
    This class has additional methods for data cleaning and manipulation,
    and adds custom properties such as `agg_vals`, `eod_capital`, `eod_pal`,
    and `bod_capital`.

    Attributes
    ----------
    agg_vals : BlotterDataFrame
        A custom DataFrame to store the aggregated values.
    eod_capital : BlotterSeries
        A custom pandas Series representing the total daily end-of-day (EOD)
        capital of the entire fund.
    eod_pal : BlotterSeries
        A custom pandas Series representing the total daily EOD profit and loss
        (PAL) capital of the entire fund.
    bod_capital : BlotterSeries
        A custom pandas Series representing the beginning-of-day (BOD)
        capital of the entire fund.

    Methods
    -------
    normalize_date_col()
        Normalize the 'date' column of the DataFrame.
    group_and_aggregate(other, params_group, params_agg)
        Group and aggregate the DataFrame based on given parameters.
    compute_eod_capital()
        Compute the total daily EOD capital of the entire fund.
    compute_eod_pal()
        Compute the total daily PAL capital of the entire fund.
    compute_bod_capital()
        Compute the BOD capital of the entire fund.
    compute_daily_return()
        Compute the daily return of the fund and add it as a new column in the DataFrame.
    replace_val_in_column(column_name, str_to_replace, update_str)
        Replace a given value in a given column of the DataFrame with a new value.
    normalize_text_col(series_name)
        Normalize a text column of the DataFrame.
    data_cleanup()
        Clean up the data in the BlotterDataFrame by normalizing date and text columns.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agg_vals: BlotterDataFrame = None
        self.eod_capital: BlotterSeries = None
        self.eod_pal: BlotterSeries = None
        self.bod_capital: BlotterSeries = None
        self._validate_req_headers(Parameters.REQ_COL_NAMES)

    @property
    def _constructor(self):
        self._normalize_headers()
        return BlotterDataFrame

    @property
    def _constructor_sliced(self):
        return BlotterSeries

    def _normalize_headers(self):
        self.columns = map(u.StringUtils.strip_lower, self.columns.to_list())

    def _validate_req_headers(self, req_cols: list) -> None:
        is_subset: bool = all(col in self.columns for col in req_cols)
        if not is_subset:
            warnings.warn(wm.requiredColumnsNotPresentWarning(req_cols, self.columns))

    @u.DecoratorUtils.check_col_exists('date')
    def normalize_date_col(self) -> None:
        """
        Normalize the `date` column of the BlotterDataFrame by removing any
        malformed dates.
        """
        self["date"] = self["date"].normalize_dates()
    
    @classmethod
    def group_and_aggregate(cls, other, params_group: list, params_agg: dict):
        """
        Group a BlotterDataFrame by specified columns and aggregate the data.

        Parameters
        ----------
        cls : BlotterDataFrame
            The class object.
        other : BlotterDataFrame
            The BlotterDataFrame to group and aggregate.
        params_group : list
            A list of column names to group the data by.
        params_agg : dict
            A dictionary where keys are column names 
            and values are functions to apply to each group.
            See pandas.DataFrame.agg for more information.
        Returns
        -------
        BlotterDataFrame
        """
        warnings.warn(wm.createNewInstanceWarninig(cls))
        return other.groupby(params_group).agg(params_agg).reset_index()

    @u.DecoratorUtils.check_col_exists('exposure')
    def compute_eod_capital(self) -> BlotterSeries:
        """
        Compute the total daily end of day (EOD) capital of the entire fund.

        Returns
        -------
        BlotterSeries
            A Pandas Series object containing the total daily EOD capital of the entire fund, 
            indexed by date.
        """
        self.eod_capital = self.groupby('date')['exposure'].sum()
        return self.eod_capital

    @u.DecoratorUtils.check_col_exists('pal')
    def compute_eod_pal(self) -> BlotterSeries:
        """
        Compute the total daily PAL capital of the entire fund.

        Returns
        -------
        BlotterSeries
            A BlotterSeries object containing the total daily PAL capital.
        """
        self.eod_pal = self.groupby('date')['pal'].sum()
        return self.eod_pal

    @u.DecoratorUtils.check_arg_notnull('eod_capital')
    @u.DecoratorUtils.check_arg_notnull('eod_pal')
    def compute_bod_capital(self) -> BlotterSeries:
        """
        Computes the beginning-of-day (BOD) capital of the entire fund.
        The BOD capital for a given day is defined as the end-of-day (EOD) capital 
        for the previous day, adjusted by the previous day's profit and loss (P&L).

        Returns
        -------
        BlotterSeries
            A new BlotterSeries object containing the BOD c
            apital values for each day.
        """
        eod_first = self.eod_capital.iloc[0] - self.eod_pal.iloc[0]
        eod_capital_shift = self.eod_capital.shift(1)
        eod_capital_shift.iloc[0] = eod_first
        self.bod_capital = eod_capital_shift.reset_index()
        return eod_capital_shift

    @u.DecoratorUtils.check_arg_notnull('bod_capital')
    def compute_daily_return(self):
        """
        Calculate daily returns based on the BlotterDataFrame's `bod_capital` arg.
        The `bod_capital` arg is used to calculate the daily returns for the fund.
        A new column named "return" is created and added to the BlotterDataFrame.

        Returns
        -------
        None
        """
        warnings.warn("creating new column return")
        self["return"] = self.apply(u.MathUtils.calculate_return,
                                    args=(self.bod_capital,), axis=1)

    def replace_val_in_column(self, column_name: str, str_to_replace: str,
                              update_str: str) -> None:
        """
        Replaces all occurrences of `str_to_replace` with `update_str` in a specified
        column of the BlotterDataFrame.

        Parameters
        ----------
        column_name : str
            The name of the column to modify.
        str_to_replace : str
            The string value to be replaced in the column.
        update_str : str
            The string value to replace `str_to_replace`.
        """
        self[column_name] = self[column_name].replace(str_to_replace, update_str)

    def normalize_text_col(self, series_name) -> None:
        """
        Normalize a text column of the BlotterDataFrame.

        Parameters
        ----------
        series_name : str
        The name of the text column to be normalized.
        """
        self[series_name] = self[series_name].normalize_texts()

    def data_cleanup(self) -> None:
        """
        Clean up the data in a BlotterDataFrame by normalizing date and text columns.

        Parameters
        ----------
        blotter : BlotterDataFrame
            The BlotterDataFrame to be cleaned up.

        Returns
        -------
        """
        self.normalize_date_col()
        for col in self.columns:
            if self[col].dtype == "object":
                self.normalize_text_col(col)


def load_csv(path: str) -> BlotterDataFrame:
    """
    Loads a CSV file from the specified path and returns a BlotterDataFrame object.
    """
    blotter_csv: pd.DataFrame = pd.read_csv(path)
    return BlotterDataFrame(blotter_csv)


def main(csv_path_in: str, csv_path_out: str):
    # absolute path to the csv
    blotter: BlotterDataFrame = load_csv(csv_path_in)
    blotter.data_cleanup()
    # we group and aggregate 
    group_by: list = ['date', 'lkid', 'sector', 'ticker', 'name']
    # we could use a lambda to manipulate sector name, but for the 
    # sake of a package, a function makes it more visible why not a function after?
    agg_operations: dict = {'analyst': 'first', 'pal': 'sum', 'exposure': 'sum'}
    # creates a new Blotter
    agg_blotter = BlotterDataFrame.group_and_aggregate(blotter, group_by, agg_operations)
    # replace naming
    agg_blotter.replace_val_in_column('sector', 'Technology', 'Information Technology')
    # computing end of day capital , sums the exposure by day
    agg_blotter.compute_eod_capital()
    # computes the end of day pal, sums pal by day, which we use for 
    # finding the exposure of the previous day
    agg_blotter.compute_eod_pal()
    # calulating the beginning of day capital
    agg_blotter.compute_bod_capital()
    # calculate the daily return for each security
    agg_blotter.compute_daily_return()
    print(f"saving csv to {csv_path_out}")
    # save the csv
    agg_blotter.to_csv(csv_path_out)
