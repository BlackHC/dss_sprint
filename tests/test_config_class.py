# Tests for config_class.
import pytest

from dss_sprint.utils.config_class import configclass


@configclass
class TestConfigClass:
    a: int
    b: str
    c: float


def test_config_class():
    config = TestConfigClass(a=1, b="2", c=3.0)
    assert config.a == 1
    assert config.b == "2"
    assert config.c == 3.0

    assert config["a"] == 1
    assert config["b"] == "2"
    assert config["c"] == 3.0


def test_config_class_conversions():
    config = TestConfigClass(a=1, b="2", c=3.0)

    assert dict(config) == {"a": 1, "b": "2", "c": 3.0}
    assert list(config) == [1, "2", 3.0]
    assert tuple(config) == (1, "2", 3.0)
    assert set(config) == {1, "2", 3.0}
    assert list(config.keys()) == ["a", "b", "c"]
    assert list(config.values()) == [1, "2", 3.0]
    assert list(config.items()) == [("a", 1), ("b", "2"), ("c", 3.0)]


def test_config_class_default_value():
    # throws exception if we don't pass c
    with pytest.raises(ValueError):
        TestConfigClass(a=1, b="2")

    TestConfigClass.c = 4.0

    config = TestConfigClass(a=1, b="2")

    assert config.c == 4.0


def test_config_class_merge_via_or():
    config = TestConfigClass(a=1, b="2", c=3.0)
    config2 = TestConfigClass(a=2, b="3", c=4.0)

    config3 = config | config2

    assert config3.a == 2
    assert config3.b == "3"
    assert config3.c == 4.0

    # test merging with a dict
    config4 = config | {"a": 3, "b": "4", "c": 5.0}

    assert config4.a == 3
    assert config4.b == "4"
    assert config4.c == 5.0





