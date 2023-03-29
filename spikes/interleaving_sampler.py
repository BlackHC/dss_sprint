"""
Spike for a dataloader that interleaves data from multiple datasets.
"""
from dataclasses import dataclass
from typing import Iterator

import torch
from torch.utils.data import DataLoader, Sampler, BatchSampler, SubsetRandomSampler, Dataset
from torch.utils.data.sampler import T_co, RandomSampler


@dataclass
class InterleavedRandomBatchSampler(Sampler[list[int]]):
    """
    Samples batches from multiple datasets.

    The first dataset is sampled without replacement, the rest are sampled with replacement.

    Each dataset has its own batchsize. The batches are concatenated.
    """
    datasets: list[Dataset]
    batch_sizes: list[int]

    def __init__(self, datasets: list[Dataset], batch_sizes: list[int]):
        super().__init__(None)
        self.datasets = datasets
        self.batch_sizes = batch_sizes

    def __iter__(self) -> Iterator[list[int]]:
        source_length = len(self.datasets[0])
        num_batches = source_length // self.batch_sizes[0]

        batch_samplers = [
            BatchSampler(RandomSampler(self.datasets[0]), batch_size=self.batch_sizes[0], drop_last=False)
        ] + [
            BatchSampler(RandomSampler(dataset, num_samples=num_batches*batch_size),
                         batch_size=batch_size, drop_last=False)
            for dataset, batch_size in zip(self.datasets[1:], self.batch_sizes[1:])
        ]
        for batches in zip(*batch_samplers):
            yield [index for batch in batches for index in batch]

    def __len__(self) -> int:
        return len(self.datasets[0]) // self.batch_sizes[0]

    def demultiplex(self, batch_tensor: list[torch.Tensor]) -> list[list[torch.Tensor]]:
        """
        Demultiplexes the batched tensor into batches for each group.
        """
        # Cumsum of batch sizes using python
        batch_sizes = [0] + self.batch_sizes
        batch_sizes = [sum(batch_sizes[:i+1]) for i in range(len(batch_sizes))]
        return [batch_tensor[batch_sizes[i]:batch_sizes[i+1]] for i in range(len(batch_sizes)-1)]

