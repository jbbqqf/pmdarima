# -*- coding: utf-8 -*-

import numpy as np
import pytest

from pmdarima.arima import utils as arima_utils
from pmdarima.compat.pytest import pytest_warning_messages, pytest_error_str


def test_issue_341():
    seas_diffed = np.array([124., -114., -163., -83.])

    with pytest.raises(ValueError) as ve:
        arima_utils.ndiffs(seas_diffed, test='adf')

    assert "raised from LinAlgError" in pytest_error_str(ve)


def test_issue_351():
    y = np.array([
        1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 1, 6, 2, 1, 0,
        2, 0, 1, 0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 3, 0, 0, 6,
        0, 0, 0, 0, 0, 1, 3, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0
    ])

    with pytest.warns(UserWarning) as w_list:
        D = arima_utils.nsdiffs(y, m=52, max_D=2, test='ocsb')

    assert D == 1

    warnings_messages = pytest_warning_messages(w_list)
    assert len(warnings_messages) == 1
    assert 'shorter than m' in warnings_messages[0]


def test_issue_586_ndiffs_returns_minimum_not_maximum():
    """Documents the contract clarified in #586: ndiffs returns the SMALLEST
    d (capped at max_d) at which the stationarity test stops requiring more
    differencing — i.e. the standard "minimum d for stationarity" you'd want
    to avoid over-differencing. The previous docstring claimed it returned
    the *maximum* such d, which would be max_d any time the series became
    stationary at any point — not what the loop actually does."""
    rng = np.random.RandomState(0)

    # Stationary series (white noise around zero): test should not ask for
    # any differencing, so d = 0. Under a "maximum d" reading we'd expect
    # max_d=2 here, which is the misreading #586 was complaining about.
    stationary = rng.randn(200)
    assert arima_utils.ndiffs(stationary, test='kpss', max_d=2) == 0
    assert arima_utils.ndiffs(stationary, test='adf', max_d=2) == 0

    # Random walk (cumulative sum of noise): non-stationary in levels,
    # stationary after one difference, so d == 1, NOT max_d.
    walk = np.cumsum(rng.randn(200))
    assert arima_utils.ndiffs(walk, test='kpss', max_d=3) == 1


def test_issue_586_nsdiffs_returns_minimum_not_maximum():
    """Companion to test_issue_586_ndiffs_returns_minimum_not_maximum — the
    same minimum-not-maximum semantics apply to seasonal differencing."""
    rng = np.random.RandomState(1)

    # Non-seasonal series → OCSB should return D = 0 (no seasonal
    # differencing required). Under the old "maximum D" reading we'd
    # expect max_D, which would clearly be wrong.
    nonseasonal = rng.randn(120)
    D = arima_utils.nsdiffs(nonseasonal, m=12, max_D=2, test='ocsb')
    assert D == 0
