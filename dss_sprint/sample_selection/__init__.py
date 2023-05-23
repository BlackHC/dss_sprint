"""
This package contains the sample selection code.

The most basic sample selection is use individual-sample acquisition functions.
"""
import typing

from dss_sprint.utils.component import Interface


class SampleSelector(typing.Protocol):
    """
    A sample selector.

    A sample selector determines the next sample(s) to be acquired.
    """

    def __call__(
        self, model: Interface, train_dataset, pool_dataset, acquisition_batch_size: int
    ) -> list[int]:
        """
        Select the next samples to be acquired.

        Args:
            model: The model.
            dataset: The dataset.
            active_learning_indices: The active learning indices.
            acquisition_batch_size: The number of samples to select.

        Returns:
            The indices of the selected samples.
        """
        ...
