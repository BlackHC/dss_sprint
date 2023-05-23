"""
Compute and use individual-sample acquisition functions.
"""
import typing
from dataclasses import dataclass

import numpy as np
import scipy

from dss_sprint.sample_selection import SampleSelector
from dss_sprint.sklearn_like_protocols import SklearnLikeClassifierProtocol
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


class EntropyScorer(IndividualScorer):
    """
    Compute the entropy of the probabilities for each pool sample.
    """

    def __call__(self, model: Interface, train_dataset, pool_dataset) -> np.ndarray:
        sklearn_like_classifier: SklearnLikeClassifierProtocol = (
            Interface.explicit_cast(SklearnLikeClassifierProtocol, model)
        )
        probs_N_C = sklearn_like_classifier.predict_proba(pool_dataset)

        # compute the entropy using numpy
        entropy_N = scipy.stats.entropy(probs_N_C, axis=-1)
        return entropy_N


class NegMaxConfidenceScorer(IndividualScorer):
    """Compute the (negative) max confidence for each pool sample"""

    def __call__(self, model: Interface, train_dataset, pool_dataset) -> np.ndarray:
        sklearn_like_classifier: SklearnLikeClassifierProtocol = (
            Interface.explicit_cast(SklearnLikeClassifierProtocol, model)
        )
        probs_N_C = sklearn_like_classifier.predict_proba(pool_dataset)

        # compute the max confidence using numpy
        neg_max_confidence_N = 1 - np.max(probs_N_C, axis=-1)
        return neg_max_confidence_N


class MarginScorer(IndividualScorer):
    """Compute the margin classifier between the top two predictions for each pool sample."""

    def __call__(self, model: Interface, train_dataset, pool_dataset) -> np.ndarray:
        sklearn_like_classifier: SklearnLikeClassifierProtocol = (
            Interface.explicit_cast(SklearnLikeClassifierProtocol, model)
        )
        probs_N_C = sklearn_like_classifier.predict_proba(pool_dataset)

        # compute the margin using numpy
        sorted_probs_N_C = np.partition(probs_N_C, kth=-2, axis=-1)[:, -2:]
        margin_N = 1 - np.abs(sorted_probs_N_C[:, 0] - sorted_probs_N_C[:, 1])
        return margin_N


class IndividualScoreBasedSelector(typing.Protocol):
    """
    Selects samples using a heuristic based on the scores of individual samples.
    """

    def __call__(self, scores_N: np.ndarray, acquisition_batch_size: int) -> list[int]:
        """
        Select samples using a heuristic based on the scores of individual samples.

        Args:
            scores_N: The scores.
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
        scores = self.scorer(model, train_dataset, pool_dataset)
        selected_indices = self.selector(scores, acquisition_batch_size)
        return selected_indices


@dataclass
class StochasticBatchSelector(IndividualScoreBasedSelector):
    coldness: float = 1.0
    mode: typing.Literal["power", "softmax", "softrank"] = "power"

    def __call__(self, scores_N: np.ndarray, acquisition_batch_size: int) -> list[int]:
        # if coldness is 0, we have random acquisition
        if self.coldness == 0.0:
            return np.random.choice(
                len(scores_N), acquisition_batch_size, replace=False
            )
        # if coldness is infinity, we have greedy acquisition
        elif np.isposinf(self.coldness):
            # select the top acquisition_batch_size samples
            return np.argsort(scores_N)[-acquisition_batch_size:][::-1]
        else:
            if self.coldness < 0:
                raise ValueError(
                    "Coldness must be positive, but is {}".format(self.coldness)
                )

        # otherwise, we have stochastic batch acquisition
        if self.mode == "softrank":
            # compute the rank of each sample and use (1/rank)**(1/coldness) as unnormed probabilities
            ranks_N = np.argsort(np.argsort(scores_N))
            unnormed_probabilities_N = (1 / (ranks_N + 1)) ** (1 / self.coldness)
            probabilities_N = unnormed_probabilities_N / np.sum(
                unnormed_probabilities_N
            )
        elif self.mode == "softmax":
            # compute the softmax of the scores / coldness using scipy.special.softmax
            probabilities_N = scipy.special.softmax(scores_N / self.coldness)
        elif self.mode == "power":
            # compute the scores / coldness and raise them to the power of coldness
            probabilities_N = (scores_N / self.coldness) ** self.coldness
            # normalize the probabilities
            probabilities_N /= np.sum(probabilities_N)
        else:
            raise ValueError(f"Unknown mode {self.mode}")
        # sample acquisition_batch_size samples from the probabilities
        return np.random.choice(
            len(scores_N), acquisition_batch_size, replace=False, p=probabilities_N
        )
