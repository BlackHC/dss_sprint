import operator
from dataclasses import dataclass
from typing import overload, Any, Iterator, Iterable, TypeVar, Optional
from collections import OrderedDict, abc as container_abcs


import skorch
import torch
import functorch


T = TypeVar('T', bound=torch.nn.Module)


class BufferList(torch.nn.Module):
    """
    A torch.nn.Module that stores a list of buffers and creates an attribute for each buffer.
    """
    def __init__(self, values: Optional[Iterable[Any]] = None) -> None:
        super(BufferList, self).__init__()
        self._size = 0
        if values is not None:
            self += values

    def _get_abs_string_index(self, idx):
        """Get the absolute index for the list of modules"""
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        return str(idx)

    @overload
    def __getitem__(self, idx: int) -> Any:
        ...

    @overload
    def __getitem__(self: T, idx: slice) -> T:
        ...

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            out = self.__class__()
            for i in range(start, stop, step):
                out.append(self[i])
            return out
        else:
            idx = self._get_abs_string_index(idx)
            return getattr(self, str(idx))

    def __setitem__(self, idx: int, param: Any) -> None:
        # Note that all other function that add an entry to the list part of
        # the ParameterList end up here. So this is the only place where we need
        # to wrap things into Parameter if needed.
        # Objects added via setattr() are not in the list part and thus won't
        # call into this function.
        idx = self._get_abs_string_index(idx)
        self.register_buffer(str(idx), param)

    def __delitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            for i in range(start, stop, step):
                del self[i]
        else:
            key = self._get_abs_string_index(key)
            delattr(self, key)
            self._size -= 1

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[Any]:
        return iter(self[i] for i in range(len(self)))

    def __iadd__(self, parameters: Iterable[Any]) -> 'BufferList':
        return self.extend(parameters)

    def __dir__(self):
        keys = super(BufferList, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def append(self, value: Any) -> 'BufferList':
        """Appends a given value at the end of the list.

        Args:
            value (Any): value to append
        """
        new_idx = len(self)
        self._size += 1
        self[new_idx] = value
        return self

    def extend(self, values: Iterable[Any]) -> 'BufferList':
        """Appends values from a Python iterable to the end of the list.

        Args:
            values (iterable): iterable of values to append
        """
        # Tensor is an iterable but we never want to unpack it here
        if not isinstance(values, container_abcs.Iterable) or isinstance(values, torch.Tensor):
            raise TypeError("BufferList.extend should be called with an "
                            "iterable, but got " + type(values).__name__)
        for value in values:
            self.append(value)
        return self

    def extra_repr(self) -> str:
        child_lines = []
        for k, p in enumerate(self):
            if isinstance(p, torch.Tensor):
                size_str = 'x'.join(str(size) for size in p.size())
                device_str = '' if not p.is_cuda else ' (GPU {})'.format(p.get_device())
                parastr = '{} containing: [{} of size {}{}]'.format(
                    "Tensor",
                    p.dtype, size_str, device_str)
                child_lines.append('  (' + str(k) + '): ' + parastr)
            else:
                child_lines.append('  (' + str(k) + '): Object of type: ' + type(p).__name__)

        tmpstr = '\n'.join(child_lines)
        return tmpstr

    def __call__(self, *args, **kwargs):
        raise RuntimeError('BufferList should not be called.')


class Parallel(torch.nn.Module):
    """
    Like a torch.nn.Sequential, but applies each module in parallel and returns
    the results as a tuple.
    """

    def __init__(self, modules):
        super().__init__()
        self.module_list = torch.nn.ModuleList(modules)

    def forward(self, x, *args, **kwargs):
        return torch.stack(tuple(m(x, *args, **kwargs) for m in self.module_list), dim=0)


class Mean(torch.nn.Module):
    """
    Take a tuple of tensors and compute the geometric average.
    """

    def forward(self, x):
        return torch.mean(torch.stack(x), dim=0)


class FunctionalModule(torch.nn.Module):
    """
    A torch.nn.Module that a function, params and buffer from functorch in a nn.Module.
    """
    func: callable
    params_list: torch.nn.ParameterList
    buffers_list: BufferList | None

    def __init__(self, func, params: torch.nn.ParameterList, buffers=None):
        super().__init__()
        self.func = func
        self.params_list = params
        self.buffers_list = buffers

    def extra_repr(self) -> str:
        return f"func={self.func.__name__}, params={self.params_list}, buffers={self.buffers_list}"

    def forward(self, x, *args, **kwargs):
        if self.buffers_list is not None:
            return self.func(list(self.params_list.parameters()), list(self.buffers_list.buffers()), x, *args, **kwargs)
        else:
            return self.func(list(self.params_list.parameters()), x, *args, **kwargs)


def make_parallel(models) -> torch.nn.Module:
    """
    Make an ensemble of models.
    """
    return Parallel(*models)


def make_functorch_parallel(models) -> FunctionalModule:
    """
    Make an ensemble of using functorch's combine_state_for_ensemble.
    """
    func, params, buffers = functorch.combine_state_for_ensemble(models)
    # convert params to a torch.nn.ParameterList
    params = torch.nn.ParameterList(
        params
    )
    # convert buffers to a BufferList
    buffers = BufferList(
        buffers
    )
    return FunctionalModule(
        func=functorch.vmap(func, in_dims=(0, 0, None)),
        params=params,
        buffers=buffers,
    )
