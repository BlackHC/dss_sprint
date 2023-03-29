import skorch
import torch
from skorch.dataset import unpack_data


class MultiObjectiveNet(skorch.NeuralNet):
    """
    A Net that has a separate iterator_outlier and supports computing an outlier exposure loss

    """
