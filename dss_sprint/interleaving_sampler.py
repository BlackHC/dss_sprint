"""
Spike for a dataloader that interleaves data from multiple datasets.
"""
import typing
from dataclasses import dataclass
from typing import Iterator

import torch
from torch.utils.data import BatchSampler, Dataset, Sampler, default_collate


# Copied and adapted from PyTorch
class SimpleRandomSampler(Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """
    data_source_length: int
    replacement: bool

    def __init__(
        self,
        data_source_length: int,
        replacement: bool = False,
        num_samples: int | None = None,
        generator=None,
    ) -> None:
        super().__init__(None)
        self.data_source_length = data_source_length
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

        if not isinstance(self.replacement, bool):
            raise TypeError(
                "replacement should be a boolean value, but got "
                "replacement={}".format(self.replacement)
            )

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return self.data_source_length
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = self.data_source_length
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(
                    high=n, size=(32,), dtype=torch.int64, generator=generator
                ).tolist()
            yield from torch.randint(
                high=n,
                size=(self.num_samples % 32,),
                dtype=torch.int64,
                generator=generator,
            ).tolist()
        else:
            for _ in range(self.num_samples // n):
                yield from torch.randperm(n, generator=generator).tolist()
            yield from torch.randperm(n, generator=generator).tolist()[
                : self.num_samples % n
            ]

    def __len__(self) -> int:
        return self.num_samples


class SimpleRandomFixedLengthSampler(Sampler):
    """
    Sometimes, you really want to do more with little data without increasing the number of epochs.

    This sampler takes a `dataset` and draws `target_length` samples from it (with repetition).
    """

    dataset_length: int
    target_length: int

    def __init__(self, dataset_length: int, target_length: int):
        super().__init__(None)
        self.dataset_length = dataset_length
        self.target_length = target_length

    def __iter__(self):
        # Ensure that we don't lose data by accident.
        if self.target_length < self.dataset_length:
            return iter(torch.randperm(self.dataset_length).tolist())

        # Sample slightly more indices to avoid biasing towards start of dataset.
        # Have the same number of duplicates for each sample.
        indices = torch.randperm(
            self.target_length + (-self.target_length % self.dataset_length)
        )

        return iter((indices[: self.target_length] % self.dataset_length).tolist())

    def __len__(self):
        return self.target_length


@dataclass
class InterleavedRandomBatchSampler(Sampler[list[int]]):
    """
    Samples batches from multiple datasets.

    The first dataset is considered 'dominant' and sampled without replacement (and drop_last=True),
    the rest are sampled with replacement. That is, the first dataset determines the epoch
    length.

    Each dataset has its own batchsize. The batches are concatenated.
    """

    training_length: int
    dataset_sizes: list[int]
    batch_sizes: list[int]
    _cum_sum_batch_sizes: list[int]
    _cum_sum_dataset_sizes: list[int]

    def __init__(
        self,
        *,
        dataset_sizes: list[int],
        batch_sizes: list[int],
        training_length: int | None = None
    ):
        super().__init__(None)
        self.dataset_sizes = dataset_sizes
        self.batch_sizes = batch_sizes

        if training_length is not None:
            self.training_length = training_length
        else:
            self.training_length = self.dataset_sizes[0]

        self._cum_sum_dataset_sizes = [
            sum(dataset_sizes[:i]) for i in range(len(dataset_sizes) + 1)
        ]
        self._cum_sum_batch_sizes = [
            sum(batch_sizes[:i]) for i in range(len(batch_sizes) + 1)
        ]

    def __iter__(self) -> Iterator[list[int]]:
        num_batches = self.training_length // self.batch_sizes[0]

        batch_samplers = [
            BatchSampler(
                SimpleRandomFixedLengthSampler(
                    self.dataset_sizes[0], self.training_length
                ),
                batch_size=self.batch_sizes[0],
                drop_last=False,
            )
        ] + [
            BatchSampler(
                SimpleRandomSampler(
                    dataset_size, num_samples=num_batches * batch_size, replacement=True
                ),
                batch_size=batch_size,
                drop_last=False,
            )
            for dataset_size, batch_size in zip(
                self.dataset_sizes[1:], self.batch_sizes[1:]
            )
        ]
        for batches in zip(*batch_samplers):
            yield [
                self._cum_sum_dataset_sizes[dataset_index] + index
                for dataset_index, batch in enumerate(batches)
                for index in batch
            ]

    def __len__(self) -> int:
        return self.dataset_sizes[0] // self.batch_sizes[0]

    def demultiplex_indices(self, batch_indices: list[int]) -> list[list[int]]:
        """
        Demultiplexes the batched tensor into batches for each group.
        """
        return [
            [
                index - self._cum_sum_dataset_sizes[dataset_index]
                for index in batch_indices[
                    self._cum_sum_batch_sizes[i] : self._cum_sum_batch_sizes[i + 1]
                ]
            ]
            for dataset_index, i in enumerate(range(len(self.batch_sizes)))
        ]

    def demultiplex_data(self, batch_data: list) -> list[list]:
        """
        Demultiplexes the batched tensor into batches for each group.
        """
        return [
            batch_data[self._cum_sum_batch_sizes[i] : self._cum_sum_batch_sizes[i + 1]]
            for i in range(len(self.batch_sizes))
        ]

    def collate(self, multiplexed_batch_data: list):
        # collate each group separately.
        # TODO: support a different collation function for each dataset?
        batches = self.demultiplex_data(multiplexed_batch_data)
        return [default_collate(batch) for batch in batches]
