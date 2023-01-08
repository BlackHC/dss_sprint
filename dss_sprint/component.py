"""
Component class to make it easy to create component based classes.

(This can allow for more robust code that can be iterated a lot.)
"""
import typing
from dataclasses import dataclass

import typing_extensions

T = typing.TypeVar('T')


@typing_extensions.runtime_checkable
class Interface(typing.Protocol):
    @classmethod
    def try_cast(cls: type[T], instance) -> T | None:
        if isinstance(instance, Component):
            return instance.query_protocol(cls)
        elif isinstance(instance, cls):
            return instance
        else:
            return None

    @classmethod
    def cast(cls: type[T], instance) -> T:
        view = cls.try_cast(instance)
        if view is None:
            raise TypeError(f'Cannot cast {instance} as {cls}')
        return view


@typing.runtime_checkable
class Component(Interface, typing.Protocol):
    """
    Default component implementation.
    """

    def query_protocol(self, cls: typing.Type[T]) -> T:
        if isinstance(self, cls):
            return self
        return None


@dataclass
class ComponentView(Component, typing.Generic[T]):
    """
    Adapter for an interface.
    """
    _component: T

    def query_protocol(self, cls: typing.Type[T]) -> T:
        if isinstance(self, cls):
            return self

        return self._component.query_protocol(cls)
