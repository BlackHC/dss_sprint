# Tests for the xpaths package

import pytest

from dss_sprint.xpath import xpath

xpath.path_separator = "."


def test_xpath():
    xpath.test_reset()
    with xpath.node("outer"):
        with xpath.node("inner"):
            assert xpath.metric_name("metric") == "outer.inner.metric"
        with xpath.node("inner2"):
            assert xpath.metric_name("metric") == "outer.inner2.metric"

    assert xpath.metric_name("metric") == "metric"

    # Check all metrics and paths
    assert xpath.all_metrics == {"metric", "outer.inner.metric", "outer.inner2.metric"}
    assert xpath.all_paths == {"", "outer", "outer.inner", "outer.inner2"}


def test_xpath_array():
    xpath.test_reset()
    with xpath.node("inner", as_array=True):
        assert xpath.metric_name("metric") == "inner[0].metric"
    with xpath.node("inner", as_array=True):
        assert xpath.metric_name("metric") == "inner[1].metric"


def test_xpath_missing_array():
    xpath.test_reset()
    # Checks that a warning is logged if the same sub_name is used twice without as_array=True
    with pytest.warns(UserWarning):
        with xpath.node("inner"):
            pass
        with xpath.node("inner"):
            pass

    # Check all metrics and paths
    assert xpath.all_paths == {"", "inner", "inner+1"}


def test_xpath_get_current_path():
    xpath.test_reset()
    with xpath.node("outer"):
        assert xpath.current_path == "outer"
        with xpath.node("inner"):
            assert xpath.current_path == "outer.inner"
        assert xpath.current_path == "outer"
    assert xpath.current_path == ""

    # Check all metrics and paths
    assert xpath.all_paths == {"", "outer", "outer.inner"}
