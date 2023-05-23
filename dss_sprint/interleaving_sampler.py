"""
Spike for a dataloader that interleaves data from multiple datasets.
"""
from dataclasses import dataclass
from typing import Iterator

from torch.utils.data import (
    BatchSampler,
    ConcatDataset,
    Dataset,
    Sampler,
    default_collate,
)
from torch.utils.data.sampler import RandomSampler


@dataclass
class InterleavedRandomBatchSampler(Sampler[list[int]]):
    """
    Samples batches from multiple datasets.

    The first dataset is considered 'dominant' and sampled without replacement (and drop_last=True),
    the rest are sampled with replacement. That is, the first dataset determines the epoch
    length.

    Each dataset has its own batchsize. The batches are concatenated.
    """

    datasets: list[Dataset]
    batch_sizes: list[int]
    _cum_sum_batch_sizes: list[int]
    _cum_sum_dataset_sizes: list[int]

    @property
    def multiplexed_dataset(self):
        return ConcatDataset(self.datasets)

    def __init__(self, datasets: list[Dataset], batch_sizes: list[int]):
        super().__init__(None)
        self.datasets = datasets
        self.batch_sizes = batch_sizes

        dataset_sizes = list(map(len, datasets))
        self._cum_sum_dataset_sizes = [
            sum(dataset_sizes[:i]) for i in range(len(dataset_sizes) + 1)
        ]
        self._cum_sum_batch_sizes = [
            sum(batch_sizes[:i]) for i in range(len(batch_sizes) + 1)
        ]

    def __iter__(self) -> Iterator[list[int]]:
        source_length = len(self.datasets[0])
        num_batches = source_length // self.batch_sizes[0]

        batch_samplers = [
            BatchSampler(
                RandomSampler(self.datasets[0]),
                batch_size=self.batch_sizes[0],
                drop_last=False,
            )
        ] + [
            BatchSampler(
                RandomSampler(
                    dataset, num_samples=num_batches * batch_size, replacement=True
                ),
                batch_size=batch_size,
                drop_last=False,
            )
            for dataset, batch_size in zip(self.datasets[1:], self.batch_sizes[1:])
        ]
        for batches in zip(*batch_samplers):
            yield [
                self._cum_sum_dataset_sizes[dataset_index] + index
                for dataset_index, batch in enumerate(batches)
                for index in batch
            ]

    def __len__(self) -> int:
        return len(self.datasets[0]) // self.batch_sizes[0]

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
