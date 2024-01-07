"""
Component class to make it easy to create component based classes.

(This can allow for more robust code that can be iterated a lot.)
"""
import typing
from dataclasses import dataclass

import typing_extensions

T = typing.TypeVar("T", bound="Component")


def explicit_try_cast(cls: typing.Type[T], instance) -> T | None:
    """Try to cast an instance to a protocol.

    If the instance implements the protocol, return the instance. Otherwise, return None.
    """
    if isinstance(instance, cls):
        return instance
    elif isinstance(instance, Component):
        return instance.query_protocol(cls)
    else:
        return None


def explicit_cast(cls: typing.Type[T], instance) -> T:
    """Cast an instance to a protocol.

    If the instance implements the protocol, return the instance. Otherwise, raise a TypeError.
    """
    view = explicit_try_cast(cls, instance)
    if view is None:
        raise TypeError(f"Cannot cast {instance} as {cls}")
    return view


@typing_extensions.runtime_checkable
class Interface(typing.Protocol):
    @classmethod
    def try_cast(cls: typing.Type[T], instance) -> T | None:
        """Try to cast an instance to the class.

        If the instance implements the class, return the instance. Otherwise, return None.
        """
        return explicit_try_cast(cls, instance)

    @classmethod
    def cast(cls: typing.Type[T], instance) -> T:
        """Cast an instance to the class.

        If the instance implements the class, return the instance. Otherwise, raise a TypeError.
        """
        return explicit_cast(cls, instance)


@typing_extensions.runtime_checkable
class Component(Interface, typing.Protocol):
    """
    Default component implementation.
    """

    def query_protocol(self, cls: typing.Type[T]) -> T | None:
        """Query the protocol for a component.

        If the component implements the protocol, return the component. Otherwise, return None.
        """
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

    def query_protocol(self, cls: typing.Type[T]) -> T | None:
        """Query the protocol for a component.

        If the component implements the protocol, return the component. Otherwise, return None.
        """
        if isinstance(self, cls):
            return self

        return self._component.query_protocol(cls)


@dataclass
class ProtocolWrapper(Component, typing.Generic[T]):
    """Adapter for a protocol.

    When you want to implement a different protocol for a component, you can use this class.
    """

    protocol_type: typing.Type[T]
    instance: T

    def query_protocol(self, cls: typing.Type[T]) -> T | None:
        if cls is self.protocol_type:
            return self.instance
        return None
