"""A baseline active learning experiment to understand the components better."""
import blackhc.project.script
import torch
from siren_pytorch import SirenNet
from sklearn.metrics import mean_squared_error
from skorch import NeuralNetRegressor
from torch import nn, optim
from tqdm.auto import tqdm

# noinspection PyUnresolvedReferences
import dss_sprint.utils.ipython_html_console
import wandb
from dss_sprint import ensembling
from dss_sprint.active_learning_indices import ActiveLearningIndices
from dss_sprint.datasets import regression1d
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

dataset = regression1d.Higdon()
X, Y = dataset.get_XY()

# convert to float32
X = X.astype("float32")
Y = Y.astype("float32")

# random split into train and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, Y.reshape(-1, 1), test_size=0.2, random_state=42
)

# %%

# Plot the data.
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.scatter(X_train, y_train, c="blue", label="train")
plt.scatter(X_test, y_test, c="red", label="test")

plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

# %%

# wandb.init(mode="disabled")
wandb.init(project="dss_sprint", group=__name__)

active_learning_indices = ActiveLearningIndices.from_pool_size(len(X_train))

# acquire a few samples randomly
active_learning_indices.acquire_randomly(10)

mse_scores = []

selector: SampleSelector = IndividualAcquisitionFunction(
    RegressionVarianceScorer(),
    StochasticBatchSelector(),
)

#%%

model_sanity = NeuralNetRegressor(
    SirenNet(dim_in=1, dim_hidden=50, dim_out=1, num_layers=3, w0_initial=30),
    max_epochs=50,
    lr=1e-2,
    optimizer=optim.Adam,
    criterion=nn.MSELoss,
    batch_size=32,
    # callbacks=[LRScheduler(policy=StepLR, step_size=10, gamma=0.1)],
    # train_split=predefined_split(torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))),
    verbose=1,
    device="mps",
)

model_sanity.fit(X_train, y_train)

y_pred = model_sanity.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Sanity check MSE: {mse}")

# Plot the data.
plt.figure(figsize=(10, 5))
plt.scatter(X_train, y_train, c="red", label="true")
plt.scatter(X_train, model_sanity.predict(X_train), c="blue", label="predicted")
plt.show()

#%%


def create_regression_models(num_models):
    models = [
        SirenNet(dim_in=1, dim_hidden=50, dim_out=1, num_layers=3, w0_initial=30)
        for _ in range(num_models)
    ]
    ensemble = ensembling.make_functorch_parallel(models)

    ensemble_model = SklearnMeanEnsembleModule(ensemble)
    return ensemble_model


#%%

model_sanity = SkorchRegressorEnsemble(
    create_regression_models(10),
    max_epochs=50,
    lr=1e-3,
    optimizer=optim.Adam,
    criterion=nn.MSELoss,
    batch_size=32,
    # callbacks=[LRScheduler(policy=StepLR, step_size=10, gamma=0.1)],
    # train_split=predefined_split(torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))),
    verbose=1,
    device="mps",
)

model_sanity.fit(X_train, y_train)

y_pred = model_sanity.predict(X)
mse = mean_squared_error(Y, y_pred)
print(f"Sanity check MSE: {mse}")

#%%
# Plot the data.
plt.figure(figsize=(10, 5))
plt.scatter(X_train, y_train, c="red", label="true")
plt.scatter(X_train, model_sanity.predict(X_train), c="blue", label="predicted")
plt.show()

# %%
import plotly.express as px

for i in tqdm(range(100 // 5)):
    # with wandb_custom_step("acquisition_step"):
    # Create a neural network regressor
    model = SkorchRegressorEnsemble(
        create_regression_models(10),
        max_epochs=50,
        lr=1e-2,
        optimizer=optim.Adam,
        criterion=nn.MSELoss,
        batch_size=32,
        # callbacks=[LRScheduler(policy=StepLR, step_size=10, gamma=0.1)],
        # train_split=predefined_split(torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))),
        verbose=1,
        device="mps",
    )

    # Train the neural network
    model.fit(
        X_train[active_learning_indices.training_indices],
        y_train[active_learning_indices.training_indices],
    )

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)

    # It would be good to use a loss that takes the implicit variance into account.

    # Plot the data.
    # plt.figure(figsize=(10, 5))
    # plt.scatter(X_train, y_train, c="red", label="true")
    # plt.scatter(X_train[active_learning_indices.training_indices], y_train[active_learning_indices.training_indices],
    #             c="green", label="acquired")
    # plt.scatter(X_train, model.predict(X_train), c="blue", label="predicted")
    #
    # plt.legend()
    # plt.title(f"Step {i}: MSE: {mse}")
    # plt.show()
    # Use plotly to plot the data
    fig = px.scatter(
        x=X[:, 0],
        y=[Y[:, 0], model.predict(X)[:, 0]],
        color_discrete_sequence=["black", "blue"],
        title=f"Step {i}: MSE: {mse}",
    )
    fig.add_scatter(
        x=X[active_learning_indices.training_indices, 0],
        y=Y[active_learning_indices.training_indices, 0],
        mode="markers",
        marker=dict(
            color="green",
        ),
    )
    # rename the traces
    fig.data[0].name = "true"
    fig.data[1].name = "predicted"
    fig.data[2].name = "acquired"

    fig.update_layout(showlegend=True)

    # also send to wandb
    log_metric("true_vs_predicted", fig)
    fig.show()

    select_samples = selector(
        model,
        X_train[active_learning_indices.training_indices],
        X_train[active_learning_indices.pool_indices],
        5,
    )

    base_indices = active_learning_indices.acquire(select_samples)
    log_metric("mse", mse)
    log_metric("acquired_index", base_indices[0])
    commit()

wandb.finish()
