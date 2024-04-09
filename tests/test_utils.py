import random

import numpy as np
import pandas as pd
import pytest

import blotter_transform.utils as ut

class TestStringUtils:

    def sample_strings():
        return [('  Test a   ', 'test a'),
                ('Test ', 'test'), (' ', '')]

    @pytest.mark.parametrize("s_tuple", sample_strings())
    def test_strip_lower(self, s_tuple):
        """ Ensure that strings are stripped correctly of whitespaces,
            and lowercase.
        """
        su = ut.StringUtils
        str_val, str_expect = s_tuple
        assert su.strip_lower(str_val) == str_expect


class TestMathUtils:

    @pytest.fixture
    def sample_dates(self) -> pd.Series:
        return pd.Series([20220101, 20220102,
                          20220103, 20220104])

    def test_compute_z_score(self, sample_dates):
        """ Ensure the validity of the z-score
        """
        # calculating z-score
        expected = np.abs((sample_dates - sample_dates.mean()) / sample_dates.std())
        # calling z-score func
        actual = ut.MathUtils.compute_z_score(sample_dates)
        np.testing.assert_array_equal(actual, expected)


class TestDateUtils:

    def sample_int_cast_num():
        return [(5, 5), (3.14159, 3), ('123', 123),
                ('  456 ', 456), ('not a number', pd.NaT),
                (None, pd.NaT)]

    @pytest.fixture
    def sample_fake_date(self):
        return 12345678

    @pytest.fixture
    def sample_outliers(self):
        return [2022018, 201508]

    @pytest.fixture
    def sample_dates(self):
        return pd.Series([
            20180201, 20180201, 20180201, 20180201,
            20180202, 20180202, 20180202, 20180202,
            20180205, 20180205, 20180205, 20180205,
            20180206, 20180206, 20180206, 20180206,
            20180207, 20180207, 20180207, 20180207
        ])

    @pytest.fixture
    def dates_with_outliers(self, sample_dates, sample_outliers: list):
        dates = sample_dates.copy()
        for rand_date in sample_outliers:
            position = random.randint(0, len(dates))
            dates.iloc[position] = rand_date
        return dates

    @pytest.fixture
    def dates_with_modified_first_index(self, sample_dates,
                                        sample_fake_date):
        dates = sample_dates.copy()
        dates.iloc[0] = sample_fake_date
        return dates

    @pytest.fixture
    def dates_with_modified_last_index(self, sample_dates,
                                       sample_fake_date):
        dates = sample_dates.copy()
        dates.iloc[-1] = sample_fake_date
        return dates

    @pytest.mark.parametrize("num_tuple", sample_int_cast_num())
    def test_castInt(self, num_tuple):
        """ """
        actual, expected = num_tuple
        int_type = ut.DateUtils.cast_int(actual)
        assert type(int_type) == type(expected)

    def test_get_outliers(self, dates_with_outliers, sample_outliers):
        series_outlier = ut.DateUtils.get_outliers(dates_with_outliers)
        assert set(series_outlier.to_list()) == set(sample_outliers)

    def test_replace_with_neighbor_first_elem(self, sample_fake_date,
                                              dates_with_modified_first_index):
        func_replace = ut.DateUtils.replace_with_neighbor
        outlier_idx, fixed_date = \
            func_replace(0, dates_with_modified_first_index,
                         [sample_fake_date])
        assert fixed_date != sample_fake_date

    def test_replace_with_neighbor_las_elem(self, sample_fake_date,
                                            dates_with_modified_last_index):
        func_replace = ut.DateUtils.replace_with_neighbor
        outlier_idx, fixed_date = \
            func_replace(dates_with_modified_last_index.index[-1],
                         dates_with_modified_last_index,
                         [sample_fake_date])
        assert fixed_date != sample_fake_date

    #def test_replace_with_neighbor_last_elem(self):
    #    pass

    #def test_replace_with_neighbor_consecutive_elem(self):
    #    pass

    # ...
