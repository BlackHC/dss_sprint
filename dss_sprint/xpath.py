# eXecution path
# This makes it easy to create nested paths for logging and debugging
import dataclasses
import warnings
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass
class XpathNode:
    full_name: str
    parent: "XpathNode" = None
    used_sub_names: {str, int} = dataclasses.field(default_factory=lambda: defaultdict(int))

    def create_child(self, sub_name, *, path_separator: str, as_array: bool = False) -> "XpathNode":
        expanded_sub_name = sub_name
        if as_array:
            expanded_sub_name = f"{sub_name}[{self.used_sub_names[sub_name]}]"
        else:
            # Check if the sub_name has been used before.
            # if self.used_sub_names[sub_name] > 0:
            #     raise ValueError(f"Sub name {sub_name} has already been used. Use an array instead!")
            if self.used_sub_names[sub_name] == 1:
                # Log a warning that the sub_name has been used before.
                warnings.warn(f"Sub name {sub_name} has already been used. Use as_array=True instead!")
                expanded_sub_name = f"{sub_name}+{self.used_sub_names[sub_name]}"

        # Increment the number of times the sub_name has been used.
        self.used_sub_names[sub_name] += 1

        if self.parent is None:
            full_name = expanded_sub_name
        else:
            full_name = f"{self.full_name}{path_separator}{expanded_sub_name}"

        return XpathNode(full_name, self)

    def get_metric_name(self, metric_name, *, path_separator: str):
        if self.parent is None:
            return metric_name
        else:
            return f"{self.full_name}{path_separator}{metric_name}"


class Xpaths:
    """
    Execution Paths Manager.

    A simple class to make it easy to create nested paths for logging and debugging.
    """
    path_separator = "/"
    _ROOT: XpathNode = XpathNode("")
    _current: ContextVar[XpathNode] = ContextVar("current", default=_ROOT)
    all_paths = {""}
    all_metrics = set()

    @contextmanager
    def sub_context(self, sub_name, as_array: bool = False):
        """
        Create a new sub-path.

        Args:
            sub_name: The name of the sub-path.
            as_array: If True, the sub-path will be treated as an array and the index will be appended to the name.
        """
        current = self._current.get()
        child = current.create_child(sub_name, as_array=as_array, path_separator=self.path_separator)
        self.all_paths |= {child.full_name}
        self._current.set(child)
        try:
            yield child
        finally:
            self._current.set(current)

    def get_current_path(self):
        """
        Get the current path.

        Returns:
            The current path.
        """
        return self._current.get().full_name

    def get_metric_name(self, metric_name):
        """
        Get the full metric name.

        Args:
            metric_name:

        Returns:
            The full metric name.
        """
        metric_name = self._current.get().get_metric_name(metric_name, path_separator=self.path_separator)
        self.all_metrics |= {metric_name}
        return metric_name


_Xpaths = Xpaths()

# Alias for creating a new sub-path.
node = _Xpaths.sub_context

# Alias for getting the full metric name.
metric_name = _Xpaths.get_metric_name

# Alias for getting the current path.
current_path = _Xpaths.get_current_path

# Limit exports
__all__ = ["node", "metric_name", "current_path", "Xpaths"]



