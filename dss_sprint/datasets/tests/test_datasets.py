import numpy as np
import torch

import dss_sprint.datasets.regression1d as datasets


def test_higdon():
    dataset = datasets.Higdon()
    X, Y = dataset.get_XY()
    assert isinstance(X, np.ndarray)
    assert isinstance(Y, np.ndarray)
    assert X.shape == (1000, 1)
    assert Y.shape == (1000, 1)

    torch_dataset = dataset.get_torch_dataset()
    assert torch_dataset[0][0].shape == torch.Size((1,))
    assert torch_dataset[0][1].shape == torch.Size((1,))
    assert len(torch_dataset) == 1000
