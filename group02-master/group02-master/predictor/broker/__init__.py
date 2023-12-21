import time
import datetime as dt
import os
import typing as t

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
# import sklearn as sk  # type: ignore
from sklearn import *  # type: ignore
from sklearn import preprocessing
from sklearn import metrics
from sklearn import base
from sklearn import model_selection
import joblib  # type: ignore

from .. import clean  # type: ignore
# import clean  # type: ignore

def format_unit(value: int, unit: str) -> str:
    """
    Formats a unit of time with proper pluralization.
    """
    return f"{value} {unit}{'s' if value != 1 else ''}"


def get_time_units(delta: dt.timedelta) -> tuple[int, int, int, int]:
    """
    Returns time units (days, hours, minutes, seconds) from a timedelta object.
    """
    total_seconds = int(delta.total_seconds())
    days, remainder = divmod(total_seconds, 60*60*24)
    hours, remainder = divmod(remainder, 60*60)
    minutes, seconds = divmod(remainder, 60)

    return days, hours, minutes, seconds


def format_timestamp(delta: dt.timedelta) -> str:
    """
    Formats a timedelta object as a string.
    """
    days, hours, minutes, seconds = get_time_units(delta)

    parts = []
    if days:
        parts.append(format_unit(days, 'day'))
    if hours:
        parts.append(format_unit(hours, 'hour'))
    if minutes:
        parts.append(format_unit(minutes, 'minute'))
    if seconds or not parts:
        parts.append(format_unit(seconds, 'second'))

    return ', '.join(parts)


class ETTAgent:
    def __init__(
        self,
        name: str,
        estimator: base.BaseEstimator,
        data: pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
        # custom_feature_exclude_list: list[str] = [],
        custom_feature_include_list: list[str] = [],
    ) -> None:
        '''
            Parameters
            ----------
            name : str
                Name of the agent
            estimator : sk.base.BaseEstimator
                The estimator to be used
            data : pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
                The data to be used for training and testing.
                If a pd.DataFrame is given, the data will be prepared, then split into training and testing sets.
                IF a 2-tuple is given, the data is assumed to be prepared, then it is split into training and testing sets.
                If a 4-tuple is given, the data is assumed to be split into training and testing sets, and thus no preparation is done. 
                    The last is the most preferred, because it does not introduce any randomness and one does not waste time for preparing the data.
            Returns
            -------
            None
        '''

        def prep(data, include_list: list[str] = []):
            X, y = clean.prepare_data(data)
            if include_list == []:
                X = X.drop(
                    [
                        "EndTime", "time",
                        "shiptype",
                        "SOG", "COG", "TH",
                        "Draught", "Length", "Breadth"
                    ],
                    axis=1
                )
            elif include_list != []:
                X = X[include_list]
            else:
                assert False, "This should never happen"
            return X, y

        self.name = name
        self.estimator = estimator
        self.fit_time = 0.0
        self.required_features = custom_feature_include_list

        if isinstance(data, pd.DataFrame):
            X, y = prep(data, custom_feature_include_list)
            data_input = model_selection.train_test_split(
                X, y, test_size=0.2, random_state=42)
        elif isinstance(data, tuple):
            if len(data) == 2:
                assert isinstance(data[0], pd.DataFrame) and \
                    isinstance(data[1], pd.DataFrame), \
                    "data[0] and data[1] must be pd.DataFrame"
                X, y = data
                data_input = model_selection.train_test_split(
                    X, y, test_size=0.2, random_state=42)
            elif len(data) == 4:
                assert isinstance(data[0], pd.DataFrame) and \
                    isinstance(data[1], pd.DataFrame) and \
                    isinstance(data[2], pd.Series) and \
                    isinstance(data[3], pd.Series), \
                    "data must be a 4-tuple of pd.DataFrame and pd.Series"

                data_input = data
            else:
                raise ValueError(
                    f"data must be either a pd.DataFrame, 2-tuple of pd.DataFrame, or 4-tuple of pd.DataFrame and pd.Series. However it is {type(data)}"
                )
        else:
            raise TypeError(
                f"data must be either a pd.DataFrame, 2-tuple of pd.DataFrame, or 4-tuple of pd.DataFrame and pd.Series. However it is {type(data)}"
            )

        self.X_train = data_input[0]  # training data
        self.X_test = data_input[1]    # test data
        self.y_train = data_input[2]  # target values that is "time left"
        self.y_test = data_input[3]    # target values of the test data

    # In sci-kit learn, this is called "fit"
    def train(
        self,
    ) -> 'ETTAgent':
        '''
            Parameters
            ----------
            X_train : array-like of shape (n_samples, n_features)
                Training data
            y_train : array-like of shape (n_samples,)
                Target values

            Returns
            -------
            self : object
                Returns self.
        '''
        assert self.X_train.shape[0] == self.y_train.shape[0], \
            "X_train and y_train must have the same number of rows"
        start_time: float = time.perf_counter()
        self.estimator.fit(self.X_train, self.y_train)
        end_time: float = time.perf_counter()
        self.fit_time = end_time - start_time
        return self

    # In sci-kit learn, this is called "predict"
    def test(self, X: pd.DataFrame = None) -> np.ndarray:
        '''
            Parameters
            ----------
            X_test : array-like of shape (n_samples, n_features)
                Testing data

            Returns
            -------
            y_pred : array-like of shape (n_samples,)
                Predicted target values
        '''

        if X is None:
            X = self.X_test
        
        if type(X) is not pd.DataFrame:
            raise TypeError(f"X must be a pd.DataFrame. However it is {type(X)}.")
        missing_features = []
        for feature in self.required_features:
            if feature not in X.columns:
                missing_features.append(feature)
        
        if len(missing_features) > 0:
            raise ValueError(f"Missing features: {missing_features}")

        if X is not None:
            return self.estimator.predict(X)
        else:
            return self.estimator.predict(self.X_test)

    def get_r2_score(self) -> float:
        # # The below metrics won't be used by the algorithm, however they can show actual inaccuracy in minutes, thus they are useful for humans
        # mae = metrics.mean_absolute_error(y_test, y_pred)
        # rmse = metrics.mean_squared_error(y_test, y_pred, squared=False)
        y_pred = self.test()
        return metrics.r2_score(self.y_test, y_pred)

    def get_result_dict(self) -> dict[str, t.Any]:
        y_pred = self.test()

        mae = metrics.mean_absolute_error(self.y_test, y_pred)
        rmse = metrics.mean_squared_error(self.y_test, y_pred, squared=False)
        mae_human = format_timestamp(dt.datetime.fromtimestamp(
            mae) - dt.datetime.fromtimestamp(0))
        rmse_human = format_timestamp(dt.datetime.fromtimestamp(
            rmse) - dt.datetime.fromtimestamp(0))
        
        r2 = metrics.r2_score(self.y_test, y_pred)

        return {
            'R2_Score': r2,
            'time': self.fit_time,
            'MAE 0': mae_human,
            'RMSE 0': rmse_human,
            # 'RMSE': rmse,
            # 'MAE': mae,
            'Predictions': y_pred
        }

    def dump(self, path: str, name=None) -> None:
        '''
            Parameters
            ----------
            path : str
                Path to the file where the agent will be saved
            name : str, optional
                Name of the file to be dumped, by default None. If None, the name of the agent will be used
        '''
        
        os.makedirs(f'{path}', exist_ok=True)
        joblib.dump(self, f'{path}/{self.name}.joblib')

    @staticmethod
    def load(path: str) -> 'ETTAgent': 
        return joblib.load(f'{path}')



def plot_results(
    name: str,
    results: dict,
    ett_agent: ETTAgent,
):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].scatter(ett_agent.y_test, results[name]['Predictions'],
                   color="blue", alpha=0.1, s=0.1)
    axs[0].scatter(ett_agent.y_test, ett_agent.y_test,
                   color="red", alpha=1.0, s=0.001)
    axs[0].set_title(f"Predictions vs Actual ETT values")

    axs[1].scatter(ett_agent.y_test, results[name]['Predictions'] - ett_agent.y_test,
                   color="blue", alpha=0.1, s=0.1)
    axs[1].scatter(ett_agent.y_test, [0] * len(ett_agent.y_test),
                   color="red", alpha=1.0, s=0.001)
    axs[1].set_title(f"Prediction error")
    # fig title
    fig.suptitle(f"{name}")
    plt.show()


class Broker:

    pass
