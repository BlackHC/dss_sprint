# Tests for the xpaths package

import pytest

from dss_sprint import xpath

xpath.Xpaths.path_separator = "."


def test_xpath():
    with xpath.node("outer"):
        with xpath.node("inner"):
            assert xpath.get_metric_name("metric") == "outer.inner.metric"
        with xpath.node("inner2"):
            assert xpath.get_metric_name("metric") == "outer.inner2.metric"

    assert xpath.get_metric_name("metric") == "metric"

    # Check all metrics and paths
    assert xpath._Xpaths.all_metrics == {"metric", "outer.inner.metric", "outer.inner2.metric"}
    assert xpath._Xpaths.all_paths == {"", "outer", "outer.inner", "outer.inner2"}


def test_xpath_array():
    with xpath.node("inner", as_array=True):
        assert xpath.get_metric_name("metric") == "inner[0].metric"
    with xpath.node("inner", as_array=True):
        assert xpath.get_metric_name("metric") == "inner[1].metric"


def test_xpath_missing_array():
    # Checks that a warning is logged if the same sub_name is used twice without as_array=True
    with pytest.warns(UserWarning):
        with xpath.node("inner"):
            pass
        with xpath.node("inner"):
            pass

    # Check all metrics and paths
    assert xpath._Xpaths.all_paths == {"", "inner", "inner+1"}
