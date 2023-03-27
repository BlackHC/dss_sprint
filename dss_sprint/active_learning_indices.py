"""
A wrapper class that splits a dataset into a training set and pool set according
to indices (or some other way of identifying samples).
"""
from dataclasses import dataclass

from dss_sprint.utils.component import Interface
import numpy as np


@dataclass
class ActiveLearningIndices:
    """
    A wrapper class that splits a dataset into a training set and pool set according
    to indices.
    """
    pool_indices: list[int]
    training_indices: list[int]

    @staticmethod
    def from_pool_size(pool_size: int):
        return ActiveLearningIndices(pool_indices=list(range(pool_size)), training_indices=[])

    def acquire(self, pool_based_indices: list[int]):
        """
        Acquire samples from within the pool set and move them to the training set.
        """
        self.training_indices += [self.pool_indices[i] for i in pool_based_indices]
        # Remove pool_based_indices from self.pool_indices
        self.pool_indices = [index for i, index in enumerate(self.pool_indices) if i not in pool_based_indices]

    def acquire_randomly(self, n: int, random_state=None):
        """
        Acquire n samples from the pool set and move them to the training set.
        """
        # Draw n random indices from the pool set.
        random = np.random.RandomState(random_state)
        pool_based_indices = random.choice(len(self.pool_indices), n, replace=False)
        return self.acquire(pool_based_indices)

