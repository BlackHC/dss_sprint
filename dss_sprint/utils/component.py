"""
Component class to make it easy to create component based classes.

(This can allow for more robust code that can be iterated a lot.)
"""
import typing
from dataclasses import dataclass

import typing_extensions

T = typing.TypeVar("T")


@typing_extensions.runtime_checkable
class Interface(typing.Protocol):
    @staticmethod
    def explicit_try_cast(cls: type[T], instance) -> T | None:
        if isinstance(instance, Component):
            return instance.query_protocol(cls)
        elif isinstance(instance, cls):
            return instance
        else:
            return None

    @classmethod
    def try_cast(cls: type[T], instance) -> T | None:
        return Interface.explicit_try_cast(cls, instance)

    @staticmethod
    def explicit_cast(cls: type[T], instance) -> T:
        view = Interface.explicit_try_cast(cls, instance)
        if view is None:
            raise TypeError(f"Cannot cast {instance} as {cls}")
        return view

    @classmethod
    def cast(cls: type[T], instance) -> T:
        return Interface.explicit_cast(cls, instance)


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

    When you want to implement a different interface for a component, you can use this class.

    Example
    -------

    >>> class InterfaceA(Interface):
    ...     def a(self):
    ...         raise NotImplementedError
    >>> class InterfaceB(Interface):
    ...     def a(self):
    ...         raise NotImplementedError
    >>> class ImplementationAB(InterfaceA, Component):
    ...     def a(self):
    ...         return 0
    ...     def b(self):
    ...         return 1
    ...     def query_protocol(self, cls: typing.Type[T]) -> T:
    ...         if issubclass(InterfaceB, cls):
    ...             return ImplementationAB.ViewB(self)
    ...     class ViewB(InterfaceB, ComponentView['ImplementationAB']):
    ...         def a(self):
    ...             return self._component.b()
    """

    _component: T

    def query_protocol(self, cls: typing.Type[T]) -> T:
        if isinstance(self, cls):
            return self

        return self._component.query_protocol(cls)


@dataclass
class ProtocolWrapper(Component):
    protocol_type: type[typing.Protocol]
    instance: object

    def query_protocol(self, cls: typing.Type[T]) -> T:
        if cls is self.protocol_type:
            return self.instance
        return None
