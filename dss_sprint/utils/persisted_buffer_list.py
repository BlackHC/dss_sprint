import operator
from collections import abc as container_abcs
from typing import Any, Iterable, Iterator, Optional, TypeVar, overload

import torch

T = TypeVar("T", bound=torch.nn.Module)


class PersistedBufferList(torch.nn.Module):
    """
    A torch.nn.Module that stores a list of buffers and creates an attribute for each buffer.
    """

    def __init__(self, values: Optional[Iterable[Any]] = None) -> None:
        super(PersistedBufferList, self).__init__()
        self._size = 0
        if values is not None:
            self += values

    def _get_abs_string_index(self, idx):
        """Get the absolute index for the list of modules"""
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError("index {} is out of range".format(idx))
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
            # iterate over all indices after the deleted one and move them into the
            # previous position
            for i in range(int(key) + 1, len(self)):
                setattr(self, str(i - 1), getattr(self, str(i)))
            delattr(self, str(len(self) - 1))
            self._size -= 1

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[Any]:
        return iter(self[i] for i in range(len(self)))

    def __iadd__(self, parameters: Iterable[Any]) -> "PersistedBufferList":
        return self.extend(parameters)

    def __dir__(self):
        keys = super(PersistedBufferList, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def append(self, value: Any) -> "PersistedBufferList":
        """Appends a given value at the end of the list.

        Args:
            value (Any): value to append
        """
        new_idx = len(self)
        self._size += 1
        self[new_idx] = value
        return self

    def extend(self, values: Iterable[Any]) -> "PersistedBufferList":
        """Appends values from a Python iterable to the end of the list.

        Args:
            values (iterable): iterable of values to append
        """
        # Tensor is an iterable but we never want to unpack it here
        if not isinstance(values, container_abcs.Iterable) or isinstance(
            values, torch.Tensor
        ):
            raise TypeError(
                "BufferList.extend should be called with an "
                "iterable, but got " + type(values).__name__
            )
        for value in values:
            self.append(value)
        return self

    def extra_repr(self) -> str:
        child_lines = []
        for k, p in enumerate(self):
            if isinstance(p, torch.Tensor):
                size_str = "x".join(str(size) for size in p.size())
                device_str = "" if not p.is_cuda else " (GPU {})".format(p.get_device())
                parastr = "{} containing: [{} of size {}{}]".format(
                    "Tensor", p.dtype, size_str, device_str
                )
                child_lines.append("  (" + str(k) + "): " + parastr)
            else:
                child_lines.append(
                    "  (" + str(k) + "): Object of type: " + type(p).__name__
                )

        tmpstr = "\n".join(child_lines)
        return tmpstr

    def __call__(self, *args, **kwargs):
        raise RuntimeError("BufferList should not be called.")
