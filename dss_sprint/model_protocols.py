"""
Model protocols.
"""
import typing

import numpy as np

from dss_sprint.component import Component

# TODO
Data: typing.TypeAlias = tuple[np.typing.ArrayLike, np.typing.ArrayLike] | np.ndarray


class TrainableModelProtocol(typing.Protocol):
    def fit(self, data: Data):
        ...


class RegressorProtocol(typing.Protocol):
    def predict(self, data: Data) -> typing.Any:
        ...


class ClassifierProtocol(typing.Protocol):
    def predict(self, data: Data) -> typing.Any:
        ...



