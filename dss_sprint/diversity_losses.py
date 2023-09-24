"""Diversity Losses for Regression and Classification."""
import functools

import skorch
import torch

from dss_sprint.interleaving_sampler import InterleavedRandomBatchSampler
from dss_sprint.sklearn_ensembling import SkorchRegressorEnsemble


class SkorchDiversityRegressorEnsemble(SkorchRegressorEnsemble):
    def __init__(self, module, *, iterator_train, loss_lambda: float, **kwargs):
        super().__init__(module, **kwargs)
        self.iterator_train = iterator_train
        self.loss_lambda = loss_lambda
        self.interleaved_batch_sampler: InterleavedRandomBatchSampler | None = None

    @functools.wraps(SkorchRegressorEnsemble.get_loss)
    def get_loss(self, y_pred, y_true, X=None, training=False):
        if not training:
            return super().get_loss(y_pred, y_true, X, training)

        # During training, we have a special interleaved sampler.
        batch_sampler: InterleavedRandomBatchSampler = self.interleaved_batch_sampler
        assert batch_sampler is not None

        # Demultiplex the batch.
        train_X, unlabeled_X = batch_sampler.demultiplex_data(X)

        (
            train_predictions_batch,
            unlabeled_predictions_batch,
        ) = batch_sampler.demultiplex_data(y_pred[1].permute(1, 0, 2))

        train_target_batch, _ = batch_sampler.demultiplex_data(y_true)
        train_loss = 0
        for batch in train_predictions_batch.permute(1, 0, 2):
            train_loss += super().get_loss(
                (batch, None), train_target_batch, train_X, training
            )

        # Apply non-linearity to unlabeled predictions.
        nonlin = self._get_predict_nonlinearity()
        unlabeled_predictions_batch: torch.Tensor = nonlin(unlabeled_predictions_batch)

        # Compute covariance matrix of unlabeled predictions using torch.
        unlabeled_covariance = unlabeled_predictions_batch[:, :, 0].cov()
        # Compute eigenvalues of unlabeled covariance matrix.
        unlabeled_eigenvalues = torch.linalg.eigvalsh(unlabeled_covariance.cpu())
        # Compute diversity loss as the sum of the log eigenvalues.
        diversity_loss = (
            torch.sum(torch.log(unlabeled_eigenvalues[unlabeled_eigenvalues > 0])) / 2
        )
        # If diversity loss is nan, set it to zero.
        if torch.isnan(diversity_loss):
            diversity_loss = 0
            # Print a warning
            print("Warning: Diversity loss is nan.")
        # Compute the total loss as the sum of the train loss and the diversity loss.
        total_loss = train_loss - diversity_loss * self.loss_lambda
        return total_loss
