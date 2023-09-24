import functorch
import torch

from dss_sprint.utils.persisted_buffer_list import PersistedBufferList


class Parallel(torch.nn.Module):
    """
    Like a torch.nn.Sequential, but applies each module in parallel and returns
    the results as a tuple.
    """

    def __init__(self, modules):
        super().__init__()
        self.module_list = torch.nn.ModuleList(modules)

    def forward(self, x, *args, **kwargs):
        return torch.stack(
            tuple(m(x, *args, **kwargs) for m in self.module_list), dim=0
        )


class Mean(torch.nn.Module):
    """
    Take a tuple of tensors and compute the geometric average.
    """

    def forward(self, x):
        return torch.mean(torch.stack(x), dim=0)


class FunctionalModule(torch.nn.Module):
    """
    A torch.nn.Module that wraps a function, params and buffer from functorch in a nn.Module.
    """

    func: callable
    params_list: torch.nn.ParameterList
    buffers_list: PersistedBufferList | None

    def __init__(self, func, params: torch.nn.ParameterList, buffers=None):
        super().__init__()
        self.func = func
        self.params_list = params
        self.buffers_list = buffers

    def extra_repr(self) -> str:
        return f"func={self.func.__name__}, params={self.params_list}, buffers={self.buffers_list}"

    def forward(self, x, *args, **kwargs):
        if self.buffers_list is not None:
            return self.func(
                list(self.params_list.parameters()),
                list(self.buffers_list.buffers()),
                x,
                *args,
                **kwargs,
            )
        else:
            return self.func(list(self.params_list.parameters()), x, *args, **kwargs)


def make_parallel(models) -> torch.nn.Module:
    """
    Make an ensemble of models.
    """
    return Parallel(models)


def make_functorch_parallel(models) -> FunctionalModule:
    """
    Make an ensemble using functorch's combine_state_for_ensemble.
    """
    func, params, buffers = functorch.combine_state_for_ensemble(models)
    # convert params to a torch.nn.ParameterList
    params = torch.nn.ParameterList(params)
    # convert buffers to a BufferList
    buffers = PersistedBufferList(buffers)
    return FunctionalModule(
        func=functorch.vmap(func, in_dims=(0, 0, None)),
        params=params,
        buffers=buffers,
    )
