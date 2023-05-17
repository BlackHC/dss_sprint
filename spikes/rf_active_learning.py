# Create a toy dataset using scikit-learn's make_classification function.
import blackhc.project.script
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification

from dss_sprint.utils.wandb_xpath import log_metric, wandb_custom_step

X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    n_clusters_per_class=2,
    weights=None,
    flip_y=0.0,
    class_sep=1.0,
    hypercube=True,
    shift=0.0,
    random_state=42,
)

#%%
# Plot the data.
plt.figure(figsize=(10, 5))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

#%%
# Split the dataset into train and test sets.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit a logistic regression model to the training data.
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model on the test set.
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))

#%%
# Plot the decision boundary.
plt.figure(figsize=(10, 5))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
x1 = np.linspace(-1, 1, 100)
x2 = -(model.intercept_ + model.coef_[0, 0] * x1) / model.coef_[0, 1]
plt.plot(x1, x2, c="k")
plt.show()

from tqdm.auto import tqdm

import wandb

#%%
from dss_sprint.active_learning_indices import ActiveLearningIndices

wandb.init(project="dss_sprint")


active_learning_indices = ActiveLearningIndices.from_pool_size(len(X_train))

# acquire a few samples randomly
active_learning_indices.acquire_randomly(10)

accuracy_scores = []

for i in tqdm(range(100)):
    with wandb_custom_step("acquisition_step"):
        model = LogisticRegression()
        model.fit(
            X_train[active_learning_indices.training_indices],
            y_train[active_learning_indices.training_indices],
        )

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)

        base_indices = active_learning_indices.acquire_randomly(1)
        log_metric("accuracy", accuracy)
        log_metric("acquired_index", base_indices[0])

#%%

# Plot accuracy scores
plt.figure(figsize=(10, 5))
plt.plot(accuracy_scores)
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.show()
