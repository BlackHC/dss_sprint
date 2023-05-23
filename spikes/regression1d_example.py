import matplotlib.pyplot as plt
import numpy as np
import skorch
import skorch.toy
import torch
import torch.nn as nn
import torch.optim as optim
from siren_pytorch import SirenNet
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from skorch import NeuralNetRegressor
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
from torch.optim.lr_scheduler import StepLR

from dss_sprint.datasets.regression1d import Higdon

#%%
# Create a dataset
dataset = Higdon()
X, y = dataset.get_XY()
X = X.astype(np.float32)
y = y.astype(np.float32)
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Sort the test data
sorted_indices = np.argsort(X_test[:, 0])
X_test = X_test[sorted_indices]
y_test = y_test[sorted_indices]


#%%

# Create a neural network regressor
net = NeuralNetRegressor(
    # skorch.toy.make_regressor(
    #     input_units=1,
    #     output_units=1,
    #     hidden_units=1000,
    #     dropout=0.2,
    #     num_hidden=5),
    SirenNet(dim_in=1, dim_hidden=50, dim_out=1, num_layers=3, w0_initial=1),
    max_epochs=100,
    lr=1e-3,
    optimizer=optim.Adam,
    criterion=nn.MSELoss,
    batch_size=32,
    # callbacks=[LRScheduler(policy=StepLR, step_size=10, gamma=0.1)],
    # train_split=predefined_split(torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))),
    verbose=1,
    device="mps",
)

# Train the neural network
net.fit(X_train, y_train)

#%%
# Plot the loss
plt.plot(net.history[:, "train_loss"], label="train")
plt.plot(net.history[:, "valid_loss"], label="test")
plt.legend()
plt.show()

#%%

# Plot the predictions
plt.scatter(X_train, y_train, label="train")
plt.scatter(X_test, net.predict(X_test), label="test", c="C1")
plt.scatter(X_test, y_test, label="test", c="C2")
plt.plot(X_test, net.predict(X_test), label="prediction", c="C1")
plt.legend()
plt.show()

#%%

# Plot the learning rate
plt.plot(net.history[:, "lr"])
plt.show()
