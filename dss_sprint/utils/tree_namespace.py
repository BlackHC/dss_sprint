import re
import typing
from typing import Any
import prettyprinter


class TreeNamespace:
    """A namespace for tree dicts.

    It stores a flat dictionary of key paths that map to values.
    Key paths are keys separated by / as strings.

    Every path can be accessed as an attribute, e.g. namespace.a.b.c for the key "a/b/c".
    Paths can also be accessed with the __getitem__ method, e.g. namespace["a/b/c"].

    A path that does not point to a value returns a new namespace with all keys that start
    with the path. For example, namespace["a"] returns a new namespace with all keys that
    start with "a/".

    The __getitem__ method also supports wildcards: namespace["a/*"] returns a new namespace
    with all keys that start with "a/". The wildcard * can be used to match any single key.
    The wildcard @ can be used to match any sub-path. For example, namespace["a/@/b"] returns
    a new namespace with all keys that start with "a/" and end with "/b".
    """

    def __init__(self, wrapped_dict):
        self._dict = wrapped_dict

    def __getattribute__(self, name: str) -> typing.Any:
        if name.startswith("_"):
            return super().__getattribute__(name)
        return self[name]

    def _schema(self) -> "TreeNamespace":
        """Get the schema of the namespace, which has the same keys
        but maps to the generic types."""
        schema_dict = {}
        for key, value in self._dict.items():
            if isinstance(value, TreeNamespace):
                sub_schema = value._schema()._dict
                for sub_key, sub_value in sub_schema.items():
                    schema_dict[key + "/" + sub_key] = sub_value
            else:
                schema_dict[key] = get_generic_type(value)
        return TreeNamespace(schema_dict)

    def _keys(self):
        """Get the direct keys of the namespace as list."""
        # Note: sets are not ordered, but dicts are ordered in Python 3.7+
        keys = {key.split("/")[0]: None for key in self._dict.keys()}
        return list(keys.keys())

    def _values(self):
        """Get the direct values of the namespace as list.

        Sub-namespaces are of type TreeNamespace, too.
        """
        return [self[key] for key in self._keys()]

    def __dir__(self) -> typing.Iterable[str]:
        return super().__dir__() + self._keys()

    def __getitem__(self, name: str) -> typing.Any:
        """Get a value from the namespace.

        * can be used as a wildcard to match any single key.
        @ can be used as a wildcard to match any sub-path.

        The returned value is either a single value or a new namespace if wild-cards are used.
        A single value can be a namespace itself if the key points to a sub-namespace.
        """
        name = str(name)
        wrapped_dict = self._dict
        if name in wrapped_dict:
            return wrapped_dict[name]

        # Check if there are any * in the name. If so, return a list of all
        # matching keys in a new namespace that only contains the matches within the
        # the key names.
        if "*" in name or "@" in name:
            # turn name into a regex: replace * with (.*) and it has to end with an optional /
            name_pattern = (
                "^"
                + name.replace("*", "([^/]*)").replace("@", "(.*?)")
                + "(?P<remainder>/.*|$)"
            )
            name_re = re.compile(name_pattern)

            # find all keys and their matches
            key_map = {}
            for key in wrapped_dict.keys():
                match = name_re.match(key)
                if not match:
                    continue

                new_parts = [group.replace("/", "_") for group in match.groups()[:-1]]
                remainder = match.groupdict()["remainder"]
                key_map[key] = "/".join(new_parts) + remainder
        else:
            # Find all keys that start with "name/" and return them as a namespace.
            # This is useful for accessing nested keys.
            key_map = {
                key: key[len(name) + 1 :]
                for key in wrapped_dict.keys()
                if key.startswith(name + "/")
            }
        if len(key_map) == 0:
            raise KeyError(f"Key '{name}' not found in {set(wrapped_dict.keys())}")
        return TreeNamespace(
            {mapped_key: wrapped_dict[key] for key, mapped_key in key_map.items()}
        )

    def __simple__repr__(self):
        if len(self._keys()) == 0:
            return {}
        elif (keys := self._keys())[0] == "0" and keys == list(
            map(str, range(len(keys)))
        ):
            return [
                value.__simple__repr__() if isinstance(value, TreeNamespace) else value
                for value in self._values()
            ]
        else:
            return {
                key: self[key].__simple__repr__()
                if isinstance(self[key], TreeNamespace)
                else self[key]
                for key in self._keys()
            }

    def __repr__(self):
        return f"DeepKeyNamespace({self.__simple__repr__()})"

    def __len__(self):
        """Get the number of keys without / or the number of unique prefixes."""
        return len(self._keys())


def get_generic_type(obj: Any) -> type:
    """Get the generic type of an object.

    For lists, tuples, and dicts, this returns the type of the elements.
    """
    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            return type(obj)
        test_type = get_generic_type(obj[0])
        if any(get_generic_type(elem) != test_type for elem in obj):
            return type(obj)
        return type(obj)[test_type]
    if isinstance(obj, dict):
        if len(obj) == 0:
            return type(obj)
        test_key_type = get_generic_type(next(iter(obj.keys())))
        test_value_type = get_generic_type(next(iter(obj.values())))
        if any(get_generic_type(key) != test_key_type for key in obj.keys()) or any(
            get_generic_type(value) != test_value_type for value in obj.values()
        ):
            return type(obj)
        return type(obj)[test_key_type, test_value_type]
    return type(obj)


@prettyprinter.register_pretty(TreeNamespace)
def pretty_deep_key_namespace(value, ctx):
    return prettyprinter.pretty_call(
        ctx,
        TreeNamespace,
        **value.__simple__repr__(),
    )
