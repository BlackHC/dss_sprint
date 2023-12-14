import threading
import types
import contextlib
import typing
from dataclasses import dataclass


class Context(types.SimpleNamespace):
    """
    A simple namespace that allows you to access attributes as if they were keys.
    """


@dataclass
class Instrumentation:
    """A simple helper that allows you to easily add instrumentation to your code.

    Example usage:

        ```
        from utils.instrumentation import instrument

        with instrument.collect() as run_info:
            # your code here
            instrument.record("loss", loss)

            with instrument("my_scope"):
                # your code here
                instrument.record("info", info)
        ```

        You can also use the `instrument` function as a decorator:

        ```
        @instrument("my_scope")
        def my_func():
            # your code here
            instrument.record("info", info)
        ```

        All records and scopes are stored as lists of SimpleNamespace objects
    """

    # Thead local storage
    _thread_local: threading.local = threading.local()

    @property
    def _current(self) -> list[Context] | None:
        """Get the current instrumentation data"""
        return getattr(self._thread_local, "current", None)

    @contextlib.contextmanager
    def scope(self, name: str):
        """Create a new scope.

        If no data is being collected currently, this is a no-op.
        """
        outer = self._current
        if outer is None:
            yield
        else:
            inner = Context()
            self._thread_local.current = inner
            try:
                yield
                outer.__dict__.setdefault(name, []).append(inner)
            finally:
                self._thread_local.current = outer

    def record(self, name: str, value: typing.Any):
        """Record a value.

        If no data is being collected currently, this is a no-op.
        """
        current = self._current
        if current is not None:
            current.__dict__.setdefault(name, []).append(value)

    @contextlib.contextmanager
    def collect(self) -> Context:
        """Collect instrumentation data.

        This is a context manager that collects all instrumentation data
        that is recorded inside the context.
        """
        outer = self._current
        scope = Context()
        self._thread_local.current = scope
        try:
            yield scope
            # If there is an outer, merge scope into it.
            if outer is not None:
                # Merge each key separately.
                for key, value in scope.__dict__.items():
                    outer.__dict__.setdefault(key, []).extend(value)
        finally:
            self._thread_local.current = outer

    def replay(self, context: Context) -> typing.Iterable[tuple[str, typing.Any]]:
        """Replay the collected instrumentation data.

        This returns an iterable of (name, value) pairs.
        """
        for key, values in context.__dict__.items():
            for value in values:
                yield key, value

    def deep_replay(
        self, context: Context
    ) -> typing.Iterable[tuple[tuple[str], typing.Any]]:
        """Replay the collected instrumentation data.

        This returns an iterable of (path, value) pairs, where path is a tuple of keys.
        """
        for key, values in context.__dict__.items():
            for value in values:
                yield (key,), value

                if isinstance(value, Context):
                    for path, value in self.deep_replay(value):
                        yield (key,) + path, value
