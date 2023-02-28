"""
Turns any callable into a configclass (which we can inherit from etc).

configfunc wraps the function in a dataclass for all default parameters and
applies configdict to it as well.
"""
import dataclasses
import inspect


def signature_apply_config(signature, config):
    """
    Combine the function signature with the config..

    Args:
        signature (inspect.Signature): The signature of the function.
        config (dict): The config to apply to the signature.

    Returns:
        A bound signature.
    """
    # Get all parameters from func and their default values.
    all_params = {k: v.default for k, v in signature.parameters.items()}
    # Update defaults of parameters with values from sub_context
    # (don't add new parameters).
    all_params.update({k: v for k, v in config.items() if k in all_params})

    # Create a new signature with the updated defaults.
    new_signature = signature.replace(
        parameters=[
            signature.parameters[k].replace(default=v) for k, v in all_params.items()
        ]
    )
    return new_signature


@dataclasses.dataclass
class ConfigFunc:
    """
    Wrapper class for config functions. We overload __setattr__, __getattr__, __delattr__ to
    forward to the wrapped functions __kwdefaults__.
    """
    __callable: callable

    def __setattr__(self, name, value):
        self.__callable.__kwdefaults__[name] = value

    def __getattr__(self, name):
        return self.__callable.__kwdefaults__[name]

    def __delattr__(self, name):
        del self.__callable.__kwdefaults__[name]

    def __call__(self, *args, **kwargs):
        return self.__callable(*args, **kwargs)




def configfunc(callable):
    # Get all default parameters from the function.
    signature = inspect.signature(callable)
    default_parameters = {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    # Get the qualified name of the function.
    name = callable.__qualname__
    # Create a dataclass with fields for the defaultparameters.
    default_parameters_dataclass = dataclasses.make_dataclass(
        name, default_parameters.keys(), slots=True, weakref_slot=True,
    )






