"""
Define a context that can be used to pass state around outside of regular parameters and instance fields.

This is useful for passing around state that is not specific to a particular instance of a class, but is
still useful to have access to. For example, the number of GPUs available on the machine can be passed
around using this context.

Importantly, we also make it possible to merge the context with external context information. This enables
dependency injection (and aspect-oriented programming).
"""
import contextlib
import dataclasses
import functools
import inspect
import types
import warnings
import typing
from collections import ChainMap
from dataclasses import dataclass
from typing import Any, Optional, TypeVar

import pytest


# @dataclass(frozen=True, slots=True)
# class ResolvedField:
#     """
#     A field that has been resolved.
#     """
#     value: Any
#     source: object  # Debug information about where the value came from.


@dataclass
class Context:
    key: object
    fields: typing.Mapping[str, Any]
    origins: typing.Mapping[str, object]

    def get(self, key: str, default=None) -> Any:
        return self.fields.get(key, default)


class ContextMergeCallback(typing.Protocol):
    def __call__(
        self, context: Context, key: object, fields: typing.Mapping[str, Any]
    ) -> Context:
        raise NotImplementedError()


@dataclass
class ContextManager:
    current_context: Optional[Context] = None

    @contextlib.contextmanager
    def context(
        self, key: object, fields: typing.Mapping[str, Any]
    ) -> typing.Iterator[Context]:
        """
        Create a new sub-context.
        """
        chain = ChainMap(
            fields, self.current_context.fields if self.current_context else {}
        )
        new_context = Context(key=key, fields=chain, origins={})

        old_context = self.current_context
        self.current_context = new_context
        try:
            yield new_context
        finally:
            self.current_context = old_context


# @init_injection(target)
# class_or_func
#
# @call_injection(target)
# func_method_etc
#
# @context
# class Context:
# 	"""
# 	Maybe @context can provide the actual context manager inside this all.
# 	"""
#
# @sub_context(class | func | method | key | special stuff)
# class SubContext:
# 	"""
# 	All other decorators inside will be conditioned on the
# 	context predict and whenever the class etc is used/called we will update the context.
# 	"""
#
# @delayed_context(frame_info)
# class SubContext:
# 	"""
# 	Whenever a new context is entered, all not yet visited stack frames will be looked to see if there are any delayed context updates to be applied.
#
# 	This is useful for more specific context updates depending on the state of the app
# 	"""
#
# class Delayed:
# 	"""
# 	Is a proxy for a field in a subcontext etc
# 	and will execute a lambda or so to resolve the value at call-time using the context and other information.
# 	"""
# 	...

"""
Design:
we have the annotation that contains the configuration.
we have the method that creates the runtime binding.
the runtime binding is a function that can evaluate whether its selector matches
"""


class TargetProtocol(typing.Protocol):
    def apply(self, context: Context) -> None:
        raise NotImplementedError()


@dataclass
class AspectAnnotation:
    """
    A marker class for aspect annotations.

    Decorators add subclasses.
    """

    targets: list[object]

    @staticmethod
    def try_get(cls_or_func) -> "AspectAnnotation | None":
        """
        Get the aspect annotation object for a class or function or None.

        Args:
            cls_or_func: class or function object.

        Returns:
            The aspect annotation object or None.
        """
        return getattr(cls_or_func, "__aspect_annotation__", None)

    @staticmethod
    def create_or_get(cls_or_func) -> "AspectAnnotation":
        """
        Add a target to the aspect annotation object for a class or function
        """
        # If the aspect annotation does not exist yet, add it to the class or function.
        annotation = AspectAnnotation.try_get(cls_or_func)
        if annotation is None:
            # Check if cls_or_func is wrapped by a decorator and try on the wrapped object if so.
            if hasattr(cls_or_func, "__wrapped__"):
                annotation = AspectAnnotation.try_get(cls_or_func.__wrapped__)
        if annotation is None:
            annotation = AspectAnnotation(targets=[])
            setattr(cls_or_func, "__aspect_annotation__", annotation)
        return annotation


def get_sub_context(cls_or_func):
    """
    Get the sub context from a class or function.

    For a class, it is all its instance variables except for magic ones.
    For a function, it is all its parameters with name and default.
    """
    if inspect.isclass(cls_or_func):
        return {k: v for k, v in cls_or_func.__dict__.items() if not k.startswith("__")}
    elif inspect.isfunction(cls_or_func):
        all_params = {
            k: v.default for k, v in inspect.signature(cls_or_func).parameters.items()
        }
        # Warn for empty defaults.
        empty_defaults = {
            k: v for k, v in all_params.items() if v is inspect.Parameter.empty
        }
        if empty_defaults:
            warnings.warn(
                f"Empty defaults for parameters {empty_defaults} in function {cls_or_func}"
            )

        # Filter out empty defaults.
        return {k: v for k, v in all_params.items() if v is not inspect.Parameter.empty}
    else:
        raise ValueError(
            f"Expected class or function, got {cls_or_func} as sub-context!"
        )


def test_get_sub_context():
    class A:
        a = 1
        b = 2

    assert get_sub_context(A) == {"a": 1, "b": 2}

    def f(a=1, b=2):
        pass

    assert get_sub_context(f) == {"a": 1, "b": 2}

    # test that we warn for empty defaults.
    def g(a, b):
        pass

    with pytest.warns(UserWarning):
        get_sub_context(g)


def patch_function(func, context_map):
    signature = inspect.signature(func)

    # Get all parameters from func and their default values.
    all_params = {k: v.default for k, v in signature.parameters.items()}
    # Update defaults of parameters with values from sub_context
    # (don't add new parameters).
    all_params.update({k: v for k, v in context_map.items() if k in all_params})

    # Create a new signature with the updated defaults.
    new_signature = signature.replace(
        parameters=[
            signature.parameters[k].replace(default=v) for k, v in all_params.items()
        ]
    )
    # We create a wrapper and apply the defaults to the args and kwargs
    # and then call the original function with the updated args and kwargs.
    # This way we can use the original function signature and still have the
    # defaults from the context.
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound_args = new_signature.bind_partial(*args, **kwargs)
        bound_args.apply_defaults()
        return func(*bound_args.args, **bound_args.kwargs)

    # Set a special marker attribute to indicate that this function is patched.
    wrapper.__aspect_patched__ = True

    return wrapper


def unpatch_function(func):
    # Remove the special marker attribute to indicate that this function is patched.
    assert func.__aspect_patched__
    # Restore the original function.
    func = func.__wrapped__
    return func


def test_patch_function_function():
    def f(a=1, b=2):
        return a, b

    assert f() == (1, 2)
    assert f(3) == (3, 2)
    assert f(3, 4) == (3, 4)

    # Patch the function.
    f = patch_function(f, {"b": 5})
    assert f() == (1, 5)
    assert f(3) == (3, 5)
    assert f(3, 4) == (3, 4)

    assert f.__aspect_patched__

    # Unpatch the function.
    f = unpatch_function(f)
    assert f() == (1, 2)
    assert f(3) == (3, 2)
    assert f(3, 4) == (3, 4)

    assert not hasattr(f, "__aspect_patched__")


def test_patch_function_kwargs():
    def f(x, /, a=1, b=2):
        return x, a, b

    assert f(0) == (0, 1, 2)
    assert f(0, 3) == (0, 3, 2)

    # Patch the function.
    f = patch_function(f, {"b": 5})
    assert f(0) == (0, 1, 5)
    assert f(0, 3) == (0, 3, 5)

    assert f.__aspect_patched__

    # Unpatch the function.
    f = unpatch_function(f)
    assert f(0) == (0, 1, 2)
    assert f(0, 3) == (0, 3, 2)

    assert not hasattr(f, "__aspect_patched__")

    # Now patch the positional argument.
    f = patch_function(f, {"x": 0})
    assert f() == (0, 1, 2)
    # TODO this is confusing as fuck.
    # We should remove positional arguments from the signature
    # or not allow them to be patched.
    assert f(a=3) == (0, 3, 2)

    assert f.__aspect_patched__

    # Unpatch the function.
    f = unpatch_function(f)
    assert f(0) == (0, 1, 2)
    assert f(0, 3) == (0, 3, 2)

    assert not hasattr(f, "__aspect_patched__")


def test_patch_function_class():
    class A:
        def __init__(self, a=1, b=2):
            self.a = a
            self.b = b

    assert A().a == 1
    assert A().b == 2

    # Patch the class.
    A.__init__ = patch_function(A.__init__, {"b": 5})
    assert A().a == 1
    assert A().b == 5

    assert A.__init__.__aspect_patched__

    # Unpatch the class.
    A.__init__ = unpatch_function(A.__init__)
    assert A().a == 1
    assert A().b == 2

    assert not hasattr(A.__init__, "__aspect_patched__")


@dataclass
class SubContextTargetDecorator:
    target: object

    @dataclass
    class Linker:
        target: object
        sub_context = object

        def __init__(self, target):
            self.target = target

        def apply(self, context_manager: ContextManager) -> None:
            # Check if target is a class or a function.
            if isinstance(self.target, type):
                # Inspect all the parameters of the __init__ method and capture them in a dict
                # Then resolve the values of the dict using the context.
                # Then call the __init__ method with the resolved values.
                signature = inspect.signature(self.target.__init__)

                # If it is a class, we need to wrap the __init__ method.
                # We monkeypatch the __init__ method to add the context manager as a parameter.
                @functools.wraps(self.target.__init__)
                def wrapped_init(self, *args, **kwargs):

                    self.__init__(*args, **kwargs)

                self.target.__init__ = wrapped_init
            elif callable(self.target):
                # If it is a function, we need to wrap the function.
                # We monkeypatch the function to add the context manager as a parameter.
                @functools.wraps(self.target)
                def wrapped_func(*args, **kwargs):
                    return self.target(*args, **kwargs)

                self.target = wrapped_func

    def __call__(self, cls_or_func):
        """Add a context target to the context."""
        annotation = AspectAnnotation.create_or_get(cls_or_func)
        annotation.targets.append(self)


SubContextTarget = SubContextTargetDecorator


@dataclass(frozen=True)
class Selector:
    parent: "Selector | None"
    target: object

    def check_parent(self, active_state: dict[object, bool]) -> bool:
        if self.parent is not None:
            return self.parent.check_active(active_state)
        return True


@dataclass
class Aspect:
    """
    Aspect class
    """


class AspectContext:
    pass


def use_configuration(config: object):
    """

    Args:
        config:

    Returns:
    """
    # walk the config and find all the @aspect annotations
    # build a
