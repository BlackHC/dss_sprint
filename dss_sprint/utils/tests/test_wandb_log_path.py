from unittest.mock import patch

import pytest

import wandb
from dss_sprint.utils.log_path import xpath
from dss_sprint.utils.wandb_log_path import (
    commit,
    define_metric,
    log,
    log_metric,
    wandb_custom_step,
)


@pytest.mark.integration
def test_integration_wandb_log_path():
    run = wandb.init(project="dummy_project", mode="online")

    # Test wandb_custom_step
    with wandb_custom_step("test_step"):
        assert xpath.current_step_name == "test_step"
        assert xpath.current_step_index == 0

        # Test log_metric
        log_metric("test_metric", 1)

        # Test log
        log({"test_metric2": 2})

        # Test commit
        commit()
        # Check if commit was successful

        # Test define_metric
        define_metric("test_metric3")

    run.finish()


def test_wandb_log_path():
    with (
        patch("wandb.log"),
        patch("wandb.define_metric"),
        patch("wandb.finish"),
        patch("wandb.init"),
    ):
        wandb.init(project="dummy_project", mode="online")

        # Test wandb_custom_step
        with wandb_custom_step("test_step"):
            assert xpath.current_step_name == "test_step"
            assert xpath.current_step_index == 0
            wandb.log.assert_called_with({"test_step/step": 0}, commit=False)

            # Test log_metric
            log_metric("test_metric", 1)
            wandb.log.assert_called_with({"test_step/test_metric": 1}, commit=False)

            # Test log
            log({"test_metric2": 2})
            wandb.log.assert_called_with({"test_step/test_metric2": 2}, commit=False)

            # Test commit
            commit()
            # Check if commit was successful
            wandb.log.assert_called_with({}, commit=True)

            # Test define_metric
            define_metric("test_metric3")
            wandb.define_metric.assert_called_once_with("test_step/test_metric3")

        wandb.log.assert_called_with({}, commit=True)

        wandb.finish()
