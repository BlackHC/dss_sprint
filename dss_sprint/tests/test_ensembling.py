import torch

from dss_sprint import ensembling


def test_parallel():
    # Test the Parallel class.
    class Model(torch.nn.Module):
        def forward(self, x):
            return x + 1

    model = Model()
    parallel = ensembling.Parallel([model, model])
    x = torch.tensor([1, 2, 3])
    y = parallel(x)
    assert torch.allclose(y[0], x + 1)
    assert torch.allclose(y[1], x + 1)
    assert len(y) == 2


def test_make_functorch_parallel():
    # Test the make_functorch_ensemble function.
    class Model(torch.nn.Module):
        """Simple MLP model."""

        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 3)
            self.linear2 = torch.nn.Linear(3, 3)

        def forward(self, x):
            return self.linear2(self.linear(x))

    model = Model()
    ensemble = ensembling.make_functorch_parallel([model, model])
    x = torch.tensor([1, 2, 3], dtype=torch.float32)
    y = ensemble(x)
    assert torch.allclose(y, ensemble(x))


def test_make_functorch_parallel_batchnorm():
    # Test the make_functorch_ensemble function.
    class Model(torch.nn.Module):
        """Simple MLP model."""

        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm1d(num_features=3, track_running_stats=False)
            self.linear = torch.nn.Linear(3, 3)
            self.linear2 = torch.nn.Linear(3, 3)

        def forward(self, x):
            return self.bn(self.linear2(self.linear(x)))

    model = Model()
    ensemble = ensembling.make_functorch_parallel([model, model])
    x = torch.tensor([[1, 2, 3], [1, 2, 3]], dtype=torch.float32)
    y = ensemble(x)
    assert torch.allclose(y, ensemble(x))


def test_make_functorch_parallel_with_buffers():
    # Test the make_functorch_ensemble function.
    class Model(torch.nn.Module):
        """Simple MLP model."""

        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 3)
            self.linear2 = torch.nn.Linear(3, 3)
            self.register_buffer("buffer", torch.tensor([1, 2, 3], dtype=torch.float32))

        def forward(self, x):
            return self.linear2(self.linear(x)) + self.buffer

    model = Model()
    ensemble = ensembling.make_functorch_parallel([model, model])
    x = torch.tensor([1, 2, 3], dtype=torch.float32)
    y = ensemble(x)
    assert torch.allclose(y, ensemble(x))
