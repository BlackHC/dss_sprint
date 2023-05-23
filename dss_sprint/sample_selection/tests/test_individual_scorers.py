import numpy as np

from dss_sprint.sample_selection.individual_scorers import StochasticBatchSelector


def test_stochastic_batch_selector_sampling():
    for mode in ["power", "softmax", "softrank"]:
        selector = StochasticBatchSelector(coldness=1.0, mode=mode)
        scores_N = np.array([0.1, 0.2, 0.3, 0.4])
        selected_indices = selector(scores_N, 2)
        assert len(selected_indices) == 2
        assert selected_indices[0] != selected_indices[1]
        assert selected_indices[0] in [0, 1, 2, 3]


def test_stochastic_batch_selector_uniform():
    for mode in ["power", "softmax", "softrank"]:
        selector = StochasticBatchSelector(coldness=0.0, mode=mode)
        scores_N = np.array([0.1, 0.2, 0.3, 0.4])
        selected_indices = selector(scores_N, 2)
        assert len(selected_indices) == 2
        assert selected_indices[0] != selected_indices[1]
        assert selected_indices[0] in [0, 1, 2, 3]


def test_stochastic_batch_selector_greedy():
    for mode in ["power", "softmax", "softrank"]:
        selector = StochasticBatchSelector(coldness=np.inf, mode=mode)
        scores_N = np.array([0.1, 0.2, 0.3, 0.4])
        selected_indices = selector(scores_N, 2)
        assert len(selected_indices) == 2
        assert list(selected_indices) == [3, 2]
