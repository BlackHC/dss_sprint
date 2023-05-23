import typing

import typing_extensions
from numpy.typing import ArrayLike


@typing.runtime_checkable
class SklearnLikeRegressorProtocol(typing.Protocol):
    """
    Protocol for a sklearn regression estimator.
    """

    def fit(self, X: ArrayLike, y: ArrayLike) -> typing_extensions.Self:
        ...

    def predict(self, X: ArrayLike) -> ArrayLike:
        ...

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        ...

    def get_params(self, deep=True) -> dict:
        ...

    def set_params(self, **params) -> typing_extensions.Self:
        ...


@typing.runtime_checkable
class SklearnLikeEnsembleRegressorProtocol(
    SklearnLikeRegressorProtocol, typing.Protocol
):
    """
    Estimator that allows drawing multiple samples using an ensemble method (either real or virtual).
    """

    n_ensemble_members_: int = ...

    def predict_all(self, X) -> ArrayLike:
        """
        Predicts multiple samples from the ensemble.

        Args:
            X:

        Returns:
            A numpy array of shape (n_samples, n_ensemble_members, n_outputs)
        """
        ...


@typing.runtime_checkable
class SklearnLikeClassifierProtocol(typing.Protocol):
    """
    Protocol for a sklearn classification estimator.
    """

    def fit(self, X: ArrayLike, y: ArrayLike) -> typing_extensions.Self:
        ...

    def predict(self, X: ArrayLike) -> ArrayLike:
        ...

    def predict_proba(self, X: ArrayLike) -> ArrayLike:
        ...

    def predict_log_proba(self, X: ArrayLike) -> ArrayLike:
        ...

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        ...

    def get_params(self, deep=True) -> dict:
        ...

    def set_params(self, **params) -> typing_extensions.Self:
        ...


@typing.runtime_checkable
class SklearnLikeEnsembleClassifierProtocol(
    SklearnLikeClassifierProtocol, typing.Protocol
):
    """
    Estimator that allows drawing multiple samples using an ensemble method (either real or virtual).
    """

    def predict_all(self, X) -> ArrayLike:
        """
        Predicts multiple samples from the ensemble.

        Args:
            X:

        Returns:
            A numpy array of shape (n_samples, n_ensemble_members)
        """
        ...

    def predict_proba_all(self, X) -> ArrayLike:
        """
        Predicts multiple samples from the ensemble.

        Args:
            X:

        Returns:
            A numpy array of shape (n_samples, n_ensemble_members, n_classes)
        """
        ...

    def predict_log_proba_all(self, X) -> ArrayLike:
        """
        Predicts multiple samples from the ensemble.

        Args:
            X:

        Returns:
            A numpy array of shape (n_samples, n_ensemble_members, n_classes)
        """
        ...
