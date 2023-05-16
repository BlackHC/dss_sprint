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


def test_buffer_list():
    # Test the ensembling.BufferList class which contains a list of buffers.
    buffer_list = ensembling.PersistedBufferList([torch.tensor([1, 2, 3])])
    assert len(buffer_list) == 1
    assert torch.allclose(buffer_list[0], torch.tensor([1, 2, 3]))
    assert torch.allclose(buffer_list[0], buffer_list[0])
    assert torch.allclose(buffer_list[0], buffer_list[0].clone())

    # Test the ensembling.BufferList class which contains a list of buffers.
    buffer_list = ensembling.PersistedBufferList(
        [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
    )
    assert len(buffer_list) == 2
    assert torch.allclose(buffer_list[0], torch.tensor([1, 2, 3]))
    assert torch.allclose(buffer_list[1], torch.tensor([4, 5, 6]))
    assert torch.allclose(buffer_list[0], buffer_list[0])
    assert torch.allclose(buffer_list[0], buffer_list[0].clone())
    assert torch.allclose(buffer_list[1], buffer_list[1])
    assert torch.allclose(buffer_list[1], buffer_list[1].clone())

    # Test an empty BufferList
    buffer_list = ensembling.PersistedBufferList([])
    assert len(buffer_list) == 0


def test_buffer_list_manual():
    bl = ensembling.PersistedBufferList()
    bl.append(torch.randn(3, 4))

    assert len(bl) == 1
    assert isinstance(bl[0], torch.Tensor)
    # Check that the buffer is registered correctly as a buffer
    assert bl._buffers["0"] is bl[0]

    bl.append(x := torch.randn(3, 4))

    # Check that the buffer is registered correctly as a buffer
    assert bl._buffers["1"] is bl[1]
    assert bl[1] is x

    # Check that we can delete buffers
    del bl[0]
    assert len(bl) == 1
    assert bl[0] is x


def test_buffer_list_in_module():
    # Test the ensembling.BufferList class and put it in a Module.
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.buffer_list = ensembling.PersistedBufferList([torch.tensor([1, 2, 3])])

        def forward(self, x):
            return x + self.buffer_list[0]

    model = Model()
    x = torch.tensor([1, 2, 3])
    y = model(x)
    assert torch.allclose(y, x + torch.tensor([1, 2, 3]))


def test_empty_buffer_list_in_module():
    # Test the ensembling.BufferList class and put it in a Module.
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.buffer_list = ensembling.PersistedBufferList()

        def forward(self, x):
            return x + len(self.buffer_list)

    model = Model()
    x = torch.tensor([1, 2, 3])
    y = model(x)
    assert torch.allclose(y, x)


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
