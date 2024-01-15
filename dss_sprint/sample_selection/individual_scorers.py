"""
Compute and use individual-sample acquisition functions.
"""
import typing
from dataclasses import dataclass

import numpy as np
import scipy

from dss_sprint.sample_selection import SampleSelector
from dss_sprint.sklearn_like_protocols import (
    SklearnLikeClassifierProtocol,
    SklearnLikeEnsembleRegressorProtocol,
)
from dss_sprint.utils.component import Interface


class IndividualScorer(typing.Protocol):
    """
    Score samples based on their probabilities.

    The higher the score, the more likely the sample is to be selected.
    That is the assumption is that higher scores are more informative.
    """

    def __call__(self, model: Interface, train_dataset, pool_dataset) -> np.ndarray:
        """
        Score samples based on their probabilities.

        Args:
            model: The model.
            train_dataset: The training dataset.
            pool_dataset: The pool dataset.

        Returns:
            The scores.
        """
        ...


class RegressionVarianceScorer(IndividualScorer):
    """
    Compute the variance of the predictions for each pool sample using an ensemble.
    """

    def __call__(self, model: Interface, train_dataset, pool_dataset) -> np.ndarray:
        regressor: SklearnLikeEnsembleRegressorProtocol = Interface.explicit_cast(
            SklearnLikeEnsembleRegressorProtocol, model
        )
        y_e_n = regressor.predict_all(pool_dataset)

        # compute the variance using numpy
        variance_n = np.var(y_e_n, axis=0)
        # verify that the variance has shape (n_samples,1)
        assert variance_n.shape == (pool_dataset.shape[0], 1)
        return variance_n[:, 0]


class EntropyScorer(IndividualScorer):
    """
    Compute the entropy of the probabilities for each pool sample.
    """

    def __call__(self, model: Interface, train_dataset, pool_dataset) -> np.ndarray:
        sklearn_like_classifier: SklearnLikeClassifierProtocol = (
            Interface.explicit_cast(SklearnLikeClassifierProtocol, model)
        )
        probs_n_c = sklearn_like_classifier.predict_proba(pool_dataset)

        # compute the entropy using numpy
        entropy_n = scipy.stats.entropy(probs_n_c, axis=-1)
        return entropy_n


class NegMaxConfidenceScorer(IndividualScorer):
    """Compute the (negative) max confidence for each pool sample"""

    def __call__(self, model: Interface, train_dataset, pool_dataset) -> np.ndarray:
        sklearn_like_classifier: SklearnLikeClassifierProtocol = (
            Interface.explicit_cast(SklearnLikeClassifierProtocol, model)
        )
        probs_n_c = sklearn_like_classifier.predict_proba(pool_dataset)

        # compute the max confidence using numpy
        neg_max_confidence_n = 1 - np.max(probs_n_c, axis=-1)
        return neg_max_confidence_n


class MarginScorer(IndividualScorer):
    """Compute the margin classifier between the top two predictions for each pool sample."""

    def __call__(self, model: Interface, train_dataset, pool_dataset) -> np.ndarray:
        sklearn_like_classifier: SklearnLikeClassifierProtocol = (
            Interface.explicit_cast(SklearnLikeClassifierProtocol, model)
        )
        probs_n_c = sklearn_like_classifier.predict_proba(pool_dataset)

        # compute the margin using numpy
        sorted_probs_n_c = np.partition(probs_n_c, kth=-2, axis=-1)[:, -2:]
        margin_n = 1 - np.abs(sorted_probs_n_c[:, 0] - sorted_probs_n_c[:, 1])
        return margin_n


class IndividualScoreBasedSelector(typing.Protocol):
    """
    Selects samples using a heuristic based on the scores of individual samples.
    """

    def __call__(self, scores_n: np.ndarray, acquisition_batch_size: int) -> list[int]:
        """
        Select samples using a heuristic based on the scores of individual samples.

        Args:
            scores_n: The scores.
            acquisition_batch_size: The number of samples to select.

        Returns:
            The indices of the selected samples.
        """
        ...


@dataclass
class IndividualAcquisitionFunction(SampleSelector):
    scorer: IndividualScorer
    selector: IndividualScoreBasedSelector

    def __call__(
        self, model: Interface, train_dataset, pool_dataset, acquisition_batch_size: int
    ) -> list[int]:
        # iterate over the pool dataset
        # compute the acquisition function for each sample
        if not len(pool_dataset):
            return []
        scores = self.scorer(model, train_dataset, pool_dataset)
        selected_indices = self.selector(scores, acquisition_batch_size)
        return selected_indices


@dataclass
class StochasticBatchSelector(IndividualScoreBasedSelector):
    coldness: float = 1.0
    mode: typing.Literal["power", "softmax", "softrank"] = "power"

    def __call__(self, scores_n: np.ndarray, acquisition_batch_size: int) -> list[int]:
        # if coldness is 0, we have random acquisition
        if self.coldness == 0.0:
            return np.random.choice(
                len(scores_n), acquisition_batch_size, replace=False
            )
        # if coldness is infinity, we have greedy acquisition
        elif np.isposinf(self.coldness):
            # select the top acquisition_batch_size samples
            return np.argsort(scores_n)[-acquisition_batch_size:][::-1]
        else:
            if self.coldness < 0:
                raise ValueError(
                    "Coldness must be positive, but is {}".format(self.coldness)
                )

        # otherwise, we have stochastic batch acquisition
        if self.mode == "softrank":
            # compute the rank of each sample and use (1/rank)**(1/coldness) as unnormed probabilities
            ranks_n = np.argsort(np.argsort(scores_n))
            unnormed_probabilities_n = (1 / (ranks_n + 1)) ** (1 / self.coldness)
            probabilities_n = unnormed_probabilities_n / np.sum(
                unnormed_probabilities_n
            )
        elif self.mode == "softmax":
            # compute the softmax of the scores / coldness using scipy.special.softmax
            probabilities_n = scipy.special.softmax(scores_n / self.coldness)
        elif self.mode == "power":
            # compute the scores / coldness and raise them to the power of coldness
            probabilities_n = (scores_n / self.coldness) ** self.coldness
            # normalize the probabilities
            probabilities_n /= np.sum(probabilities_n)
        else:
            raise ValueError(f"Unknown mode {self.mode}")
        # sample acquisition_batch_size samples from the probabilities
        return np.random.choice(
            len(scores_n), acquisition_batch_size, replace=False, p=probabilities_n
        )
