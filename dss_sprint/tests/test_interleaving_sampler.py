import torch
from torch.utils.data import TensorDataset

from dss_sprint.interleaving_sampler import InterleavedRandomBatchSampler


def test_interleaving_random_batch_sampler():
    dominant_dataset = TensorDataset(torch.linspace(0, 9, 10), torch.linspace(0, 9, 10))
    sub_dataset = TensorDataset(torch.linspace(50, 59, 10), torch.linspace(50, 59, 10))

    sampler = InterleavedRandomBatchSampler([dominant_dataset, sub_dataset], [2, 5])

    for batch in sampler:
        indices = sampler.demultiplex_indices(batch)
        # assert all indices for all groups are in range
        assert all(
            0 <= index < len(dataset)
            for dataset, indices in zip(sampler.datasets, indices)
            for index in indices
        )

        data = sampler.demultiplex_data([sampler.multiplexed_dataset[i] for i in batch])
        # assert the length of each group is correct
        assert all(
            len(group) == batch_size
            for group, batch_size in zip(data, sampler.batch_sizes)
        )
        # assert all data for all groups are in range
        assert all(0 <= data[0][j][i] < 10 for i in range(2) for j in range(2))
        assert all(50 <= data[1][j][i] < 60 for i in range(2) for j in range(5))

        fetched_data = [sampler.multiplexed_dataset[i] for i in batch]
        # test collation
        collated = sampler.collate(fetched_data)
        group1, group2 = collated
        assert len(group1) == 2
        assert len(group2) == 2

        group1_x, group1_y = group1
        group2_x, group2_y = group2

        assert group1_x.shape == (2,)
        assert group1_y.shape == (2,)

        assert group2_x.shape == (5,)
        assert group2_y.shape == (5,)

        # test that the data matches the data from above
        assert group1_x[0] == data[0][0][0]
        assert group1_x[1] == data[0][1][0]
        assert group1_y[0] == data[0][0][1]
        assert group1_y[1] == data[0][1][1]

        assert group2_x[0] == data[1][0][0]
        assert group2_x[1] == data[1][1][0]
        assert group2_x[2] == data[1][2][0]
        assert group2_x[3] == data[1][3][0]
        assert group2_x[4] == data[1][4][0]
        assert group2_y[0] == data[1][0][1]
        assert group2_y[1] == data[1][1][1]
        assert group2_y[2] == data[1][2][1]
        assert group2_y[3] == data[1][3][1]
        assert group2_y[4] == data[1][4][1]
