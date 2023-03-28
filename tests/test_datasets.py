import torch

from dss_sprint import datasets
import numpy as np


def test_higdon():
    dataset = datasets.Higdon()
    X, Y = dataset.get_XY()
    assert isinstance(X, np.ndarray)
    assert isinstance(Y, np.ndarray)
    assert X.shape == (1000,)
    assert Y.shape == (1000,)

    torch_dataset = dataset.get_torch_dataset()
    assert torch_dataset[0][0].shape == torch.Size(())
    assert torch_dataset[0][1].shape == torch.Size(())
    assert len(torch_dataset) == 1000
