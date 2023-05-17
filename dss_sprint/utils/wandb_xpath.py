import functools
from contextlib import contextmanager

import wandb
from dss_sprint.utils.xpath import xpath


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
        try:
            yield
        finally:
            wandb.log({step_metric: xpath.current_step_index}, commit=True)


def log_metric(metric_name, value, is_summary=False):
    """
    Log a metric to wandb.

    Args:
        metric_name: The metric name.
        value: The metric value.
        is_summary: Whether this is a summary metric.
    """
    metric_name = xpath.metric_name(metric_name, is_summary=is_summary)
    if is_summary is False:
        wandb.log({metric_name: value}, commit=False)
    else:
        wandb.run.summary[metric_name] = value


def log(metrics, is_summary=False):
    """
    Log metrics to wandb.

    Args:
        metrics: The metrics.
        is_summary: Whether this is a summary metric.
    """
    metrics = {
        xpath.metric_name(k, is_summary=is_summary): v for k, v in metrics.items()
    }
    if is_summary is False:
        return wandb.log(metrics, commit=False)
    else:
        return wandb.run.summary.update(metrics)


def commit():
    """
    Commit the metrics to wandb.
    """
    return wandb.log({}, commit=True)


@functools.wraps(wandb.define_metric)
def define_metric(metric_name, *args, **kwargs):
    """
    Define a metric for wandb.

    Args:
        metric_name: The metric name.
        ...: The other arguments.
    """
    metric_name = xpath.metric_name(metric_name)
    return wandb.define_metric(metric_name, *args, **kwargs)
