# -*- coding: utf-8 -*-
#
# Base classes and interfaces

import abc
from abc import ABCMeta

from sklearn.base import BaseEstimator

# TODO: change this to base TS model if we ever hope to support more


class BaseARIMA(BaseEstimator, metaclass=ABCMeta):
    """A base ARIMA class"""

    @abc.abstractmethod
    def fit(self, y, X, **fit_args):
        """Fit an ARIMA model"""

    def fit_predict(self, y, X=None, n_periods=10, **fit_args):
        """Fit an ARIMA to a vector, ``y``, of observations with an
        optional matrix of ``exogenous`` variables, and then generate
        predictions.

        Parameters
        ----------
        y : array-like or iterable, shape=(n_samples,)
            The time-series to which to fit the ``ARIMA`` estimator. This may
            either be a Pandas ``Series`` object (statsmodels can internally
            use the dates in the index), or a numpy array. This should be a
            one-dimensional array of floats, and should not contain any
            ``np.nan`` or ``np.inf`` values.

        X : array-like, shape=[n_obs, n_vars], optional (default=None)
            An optional 2-d array of exogenous variables. If provided, ``X``
            must contain rows for **both** the training observations and
            the ``n_periods`` future periods to forecast — i.e.
            ``X.shape[0] == len(y) + n_periods``. ``fit_predict`` will use
            the first ``len(y)`` rows for fitting and the last
            ``n_periods`` rows for forecasting. This should not include a
            constant or trend.

        n_periods : int, optional (default=10)
            The number of periods in the future to forecast.

        fit_args : dict or kwargs, optional (default=None)
            Any keyword args to pass to the fit method.
        """
        # Issue #514: previously the SAME `X` was passed to both fit() and
        # predict(), which raised inside predict because predict requires
        # X.shape[0] == n_periods while fit requires X.shape[0] == len(y).
        # Require the caller to provide one combined X covering both the
        # training window and the forecast horizon, and split it here. The
        # `X is None` short-circuit preserves the no-exog code path used by
        # most callers (and the existing ARIMA/AutoARIMA tests).
        X_fit, X_pred = X, X
        if X is not None:
            n_train = (
                len(y) if hasattr(y, "__len__")
                else getattr(y, "shape", (0,))[0]
            )
            expected = n_train + n_periods
            if getattr(X, "shape", (0,))[0] != expected:
                raise ValueError(
                    f"When X is provided to fit_predict, it must cover both "
                    f"the training samples and the forecast horizon: "
                    f"expected X.shape[0] == len(y) + n_periods "
                    f"({n_train} + {n_periods} = {expected}), got "
                    f"{getattr(X, 'shape', (0,))[0]}. If you only have X for "
                    f"the training window, call fit(y, X) and predict("
                    f"n_periods, X=X_future) separately."
                )
            # Use iloc when available (pandas) so the index is preserved on
            # both halves; numpy slicing falls back to the second branch.
            if hasattr(X, "iloc"):
                X_fit = X.iloc[:n_train]
                X_pred = X.iloc[n_train:]
            else:
                X_fit = X[:n_train]
                X_pred = X[n_train:]

        self.fit(y, X_fit, **fit_args)
        # fit_args is intentionally NOT forwarded to predict; predict's
        # signature only forwards extras to statsmodels' get_prediction, and
        # most fit_args (e.g. method=, start_params=) would error there.
        return self.predict(n_periods=n_periods, X=X_pred)

    # TODO: remove kwargs from all of these

    @abc.abstractmethod
    def predict(self, n_periods, X, return_conf_int=False, alpha=0.05,
                **kwargs):
        """Create forecasts on a fitted model"""

    @abc.abstractmethod
    def predict_in_sample(self, X, start, end, dynamic, **kwargs):
        """Get in-sample forecasts"""

    @abc.abstractmethod
    def update(self, y, X=None, maxiter=None, **kwargs):
        """Update an ARIMA model"""
