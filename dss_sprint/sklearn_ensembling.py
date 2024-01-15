import functools
import typing

import numpy as np
import skorch
import torch
from numpy.typing import ArrayLike
from skorch.utils import to_numpy

from dss_sprint.sklearn_like_protocols import (
    SklearnLikeClassifierProtocol,
    SklearnLikeEnsembleClassifierProtocol,
    SklearnLikeEnsembleRegressorProtocol,
    SklearnLikeRegressorProtocol,
)
from dss_sprint.utils.component import Component, T


class EnsembleOutput(typing.NamedTuple):
    """The output of an ensemble model."""
    model_predictions: ArrayLike
    ensemble_predictions: ArrayLike


class SklearnMeanEnsembleModule(torch.nn.Module):
    """
    A torch.nn.Module that wraps an ensemble model and outputs both the mean and the individual predictions.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x, *args, **kwargs):
        y_e_ = self.model(x, *args, **kwargs)
        prediction_ = torch.mean(y_e_, dim=0)
        return EnsembleOutput(prediction_, y_e_)


class SklearnLogSumExpEnsembleModule(torch.nn.Module):
    """
    A torch.nn.Module that wraps a model and outputs both the mean and the individual predictions.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x, *args, **kwargs):
        y_e_ = self.model(x, *args, **kwargs)
        prediction_ = torch.logsumexp(y_e_, dim=0)
        return EnsembleOutput(prediction_, y_e_)


class SklearnClassifierEnsemble(SklearnLikeEnsembleClassifierProtocol, Component):
    def __init__(
        self, model: skorch.NeuralNetClassifier | skorch.NeuralNetBinaryClassifier
    ):
        self.model: skorch.NeuralNetClassifier | skorch.NeuralNetBinaryClassifier = (
            model
        )

    def query_protocol(self, cls: typing.Type[T]) -> T | None:
        if isinstance(self, cls):
            return self
        elif isinstance(self.model, cls):
            return self.model
        return None

    def predict_all(self, X) -> ArrayLike:
        return self.predict_proba_all(X).argmax(axis=-1)

    def predict_proba_all(self, X) -> ArrayLike:
        nonlin = self.model._get_predict_nonlinearity()
        y_probas = []
        for yp in self.model.forward_iter(X, training=False):
            yp = yp[1]
            yp = nonlin(yp)
            y_probas.append(to_numpy(yp))
        y_proba = np.concatenate(y_probas, 0)
        return y_proba

    def _get_log_predict_nonlinearity(self):
        nonlin: typing.Callable = self.model._get_predict_nonlinearity()
        assert callable(nonlin), "nonlinearity must be callable"
        # replace torch softmax with log_softmax
        if nonlin == torch.softmax:
            return torch.log_softmax

        def log_nonlin(x):
            return torch.log(nonlin(x))

        return log_nonlin

    def predict_log_proba_all(self, X) -> ArrayLike:
        nonlin = self._get_log_predict_nonlinearity()
        y_probas = []
        for yp in self.model.forward_iter(X, training=False):
            yp = yp[1]
            yp = nonlin(yp)
            y_probas.append(to_numpy(yp))
        y_proba = np.concatenate(y_probas, 1)
        return y_proba

    def predict(self, X: ArrayLike) -> ArrayLike:
        return self.model.predict(X)

    def predict_proba(self, X: ArrayLike) -> ArrayLike:
        return self.model.predict_proba(X)

    def predict_log_proba(self, X: ArrayLike) -> ArrayLike:
        nonlin = self._get_log_predict_nonlinearity()
        y_probas = []
        for yp in self.model.forward_iter(X, training=False):
            yp = yp[0] if isinstance(yp, tuple) else yp
            yp = nonlin(yp)
            y_probas.append(to_numpy(yp))
        y_proba = np.concatenate(y_probas, 1)
        return y_proba


class SkorchRegressorEnsemble(
    skorch.NeuralNetRegressor, SklearnLikeEnsembleRegressorProtocol, Component
):
    def predict_all(self, X) -> ArrayLike:
        nonlin = self._get_predict_nonlinearity()
        ys = []
        for y_batch in self.forward_iter(X, training=False):
            assert isinstance(y_batch, EnsembleOutput)
            y_batch = y_batch.ensemble_predictions
            y_batch = nonlin(y_batch)
            ys.append(to_numpy(y_batch))
        if len(ys) >= 1:
            ys = np.concatenate(ys, 1)
        else:
            ys = np.array(ys)
        return ys

    @functools.wraps(skorch.NeuralNetRegressor.get_loss)
    def get_loss(self, y_pred, y_true, X=None, training=False):
        return super().get_loss(y_pred[0], y_true, X, training)
