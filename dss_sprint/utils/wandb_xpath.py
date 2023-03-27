import functools
from contextlib import contextmanager

from xpath import xpath
import wandb


@contextmanager
def wandb_custom_step(name):
    """
    A context manager for wandb steps.

    Args:
        name: The name of the step.
        is_summary_step: Whether this is a summary step.

    Returns:
        The context manager.
    """
    with xpath.step(name, is_summary_step=False):
        step_metric = xpath.metric_name("step", is_summary=False)
        wandb.log({step_metric: xpath.current_step_index}, commit=False)
        if xpath.current_step_count == 1:
            wandb.define_metric(f"{xpath.current_step_name}/*", step_metric)
        yield


def log_metric(metric_name, value, is_summary=False):
    """
    Log a metric to wandb.

    Args:
        metric_name: The metric name.
        value: The metric value.
        is_summary: Whether this is a summary metric.
    """
    return wandb.log({metric_name: value}, is_summary=is_summary, commit=False)


def log(metrics, is_summary=False):
    """
    Log metrics to wandb.

    Args:
        metrics: The metrics.
        is_summary: Whether this is a summary metric.
    """
    metrics = {xpath.metric_name(k, is_summary=is_summary): v for k, v in metrics.items()}
    return wandb.log(metrics, commit=False)


@functools.wraps(wandb.define_metric)
def define_metric(metric_name, *args, is_summary: bool = False, **kwargs):
    """
    Define a metric for wandb.

    Args:
        metric_name: The metric name.
        step_metric: The step metric.
        is_summary: Whether this is a summary metric.
    """
    metric_name = xpath.metric_name(metric_name, is_summary=is_summary)
    return wandb.define_metric(metric_name, *args, **kwargs)
