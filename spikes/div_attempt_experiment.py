"""A baseline active learning experiment to understand the components better."""
import blackhc.project.script
import numpy as np
import skorch
import skorch.dataset
import torch
from siren_pytorch import SirenNet
from sklearn.metrics import mean_squared_error
from skorch import NeuralNetRegressor
from torch import nn, optim
from torch.utils.data import ConcatDataset
from tqdm.auto import tqdm

# noinspection PyUnresolvedReferences
import dss_sprint.utils.ipython_html_console
import wandb
from dss_sprint import ensembling
from dss_sprint.active_learning_indices import ActiveLearningIndices
from dss_sprint.datasets import regression1d
from dss_sprint.diversity_losses import SkorchDiversityRegressorEnsemble
from dss_sprint.interleaving_sampler import InterleavedRandomBatchSampler
from dss_sprint.sample_selection import SampleSelector
from dss_sprint.sample_selection.individual_scorers import (
    EntropyScorer,
    IndividualAcquisitionFunction,
    RegressionVarianceScorer,
    StochasticBatchSelector,
)
from dss_sprint.sklearn_ensembling import (
    SklearnMeanEnsembleModule,
    SkorchRegressorEnsemble,
)
from dss_sprint.utils.wandb_log_path import commit, log_metric

dataset = regression1d.Higdon(n=128)
X_train, y_train = dataset.get_XY()
X_test, y_test = regression1d.Higdon(n=256, random_state=10).get_XY()

# %%

# Plot the data.
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.scatter(X_test, y_test, c="red", label="test", s=1)
plt.scatter(X_train, y_train, c="blue", label="train")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

#%%


def create_regression_models(num_models):
    models = [
        SirenNet(dim_in=1, dim_hidden=50, dim_out=1, num_layers=3, w0_initial=5)
        for _ in range(num_models)
    ]
    ensemble = ensembling.make_parallel(models)

    ensemble_model = SklearnMeanEnsembleModule(ensemble)
    return ensemble_model


def create_train_iterator(
    train_dataset, unlabeled_dataset, training_length, batch_size
):
    multiplexed_dataset = ConcatDataset([train_dataset, unlabeled_dataset])
    model_sanity.interleaved_batch_sampler = InterleavedRandomBatchSampler(
        dataset_sizes=[len(train_dataset), len(unlabeled_dataset)],
        batch_sizes=[batch_size, batch_size],
        training_length=training_length,
    )
    return torch.utils.data.DataLoader(
        multiplexed_dataset,
        batch_sampler=model_sanity.interleaved_batch_sampler,
        num_workers=0,
        pin_memory=True,
        # These settings sound weird but are needed together a custom batch_sampler.
        batch_size=1,
        drop_last=False,
        shuffle=False,
    )


model_sanity = SkorchDiversityRegressorEnsemble(
    create_regression_models(4),
    max_epochs=200,
    lr=1e-3,
    optimizer=optim.Adam,
    optimizer__weight_decay=1e-4,
    criterion=nn.MSELoss,
    batch_size=128,
    # callbacks=[LRScheduler(policy=StepLR, step_size=10, gamma=0.1)],
    # train_split=predefined_split(torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))),
    verbose=1,
    device="mps",
    loss_lambda=0.000001,
    iterator_train=create_train_iterator,
    iterator_train__unlabeled_dataset=skorch.dataset.Dataset(X_test, y_test),
    iterator_train__training_length=256,
)

model_sanity.fit(X_train, y_train)
# model_sanity.fit(X_train[X_train[:, 0] < 8, :], y_train[X_train[:, 0] < 8, :])

#%%

y_pred = model_sanity.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Sanity check MSE: {mse}")

test_sorted_indices = np.argsort(X_test, axis=0).reshape(-1)

# Plot the data.
plt.figure(figsize=(10, 5))
plt.scatter(X_train, y_train, c="red", label="true")
plt.plot(
    X_test[test_sorted_indices],
    model_sanity.predict(X_test[test_sorted_indices]),
    c="blue",
    label="predicted",
)
plt.show()


# Plot all ensemble members using predict_all
plt.figure(figsize=(10, 5))
plt.scatter(X_train, y_train, c="black", label="true")

for y_pred in model_sanity.predict_all(X_test):
    plt.plot(
        X_test[test_sorted_indices],
        y_pred[test_sorted_indices],
        label="predicted",
        alpha=0.5,
    )
plt.show()

#
# # %%
#
# active_learning_indices = ActiveLearningIndices.from_pool_size(len(X_train))
#
# # acquire a few samples randomly
# active_learning_indices.acquire_randomly(10)
#
# mse_scores = []
#
# selector: SampleSelector = IndividualAcquisitionFunction(
#     RegressionVarianceScorer(),
#     StochasticBatchSelector(),
# )

#
# # %%
# # wandb.init(mode="disabled")
# wandb.init(project="dss_sprint", group=__name__)
#
# import plotly.express as px
#
# for i in tqdm(range(100 // 5)):
#     # with wandb_custom_step("acquisition_step"):
#     # Create a neural network regressor
#     model = SkorchRegressorEnsemble(
#         create_regression_models(10),
#         max_epochs=50,
#         lr=1e-2,
#         optimizer=optim.Adam,
#         criterion=nn.MSELoss,
#         # callbacks=[LRScheduler(policy=StepLR, step_size=10, gamma=0.1)],
#         # train_split=predefined_split(torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))),
#         verbose=1,
#         device="mps",
#     )
#
#     # Train the neural network
#     model.fit(
#         X_train[active_learning_indices.training_indices],
#         y_train[active_learning_indices.training_indices],
#     )
#
#     y_pred = model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     mse_scores.append(mse)
#
#     # It would be good to use a loss that takes the implicit variance into account.
#
#     # Plot the data.
#     # plt.figure(figsize=(10, 5))
#     # plt.scatter(X_train, y_train, c="red", label="true")
#     # plt.scatter(X_train[active_learning_indices.training_indices], y_train[active_learning_indices.training_indices],
#     #             c="green", label="acquired")
#     # plt.scatter(X_train, model.predict(X_train), c="blue", label="predicted")
#     #
#     # plt.legend()
#     # plt.title(f"Step {i}: MSE: {mse}")
#     # plt.show()
#     # Use plotly to plot the data
#     fig = px.scatter(
#         x=X[:, 0],
#         y=[Y[:, 0], model.predict(X)[:, 0]],
#         color_discrete_sequence=["black", "blue"],
#         title=f"Step {i}: MSE: {mse}",
#     )
#     fig.add_scatter(
#         x=X[active_learning_indices.training_indices, 0],
#         y=Y[active_learning_indices.training_indices, 0],
#         mode="markers",
#         marker=dict(
#             color="green",
#         ),
#     )
#     # rename the traces
#     fig.data[0].name = "true"
#     fig.data[1].name = "predicted"
#     fig.data[2].name = "acquired"
#
#     fig.update_layout(showlegend=True)
#
#     # also send to wandb
#     log_metric("true_vs_predicted", fig)
#     fig.show()
#
#     select_samples = selector(
#         model,
#         X_train[active_learning_indices.training_indices],
#         X_train[active_learning_indices.pool_indices],
#         5,
#     )
#
#     base_indices = active_learning_indices.acquire(select_samples)
#     log_metric("mse", mse)
#     log_metric("acquired_index", base_indices[0])
#     commit()
#
# wandb.finish()
