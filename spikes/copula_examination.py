"""
# Copula Examination

We examine how to create a copula based on empirical data. We will use the
copula to generate synthetic data that is similar to the empirical data.
"""
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
import sklearn.datasets

# %%

# Create 2D dataset that has a correlated structure that is not linear
# (i.e. not a Gaussian copula)
def make_independent_dataset(n_samples=1000):
    x = np.random.uniform(0, 1, n_samples)
    y = np.random.uniform(0, 1, n_samples)
    x = np.sin(2 * np.pi * x)
    y = np.sin(2 * np.pi * y)
    return np.stack([x, y], axis=1)


# %%
# Create 2D dataset that has a correlated structure that is not linear
# (i.e. not a Gaussian copula)
def make_dataset(n_samples=1000):
    x = np.random.uniform(0, 1, n_samples)
    y = np.random.uniform(0, 1, n_samples)
    x = np.sin(2 * np.pi * x)
    y = np.sin(2 * np.pi * y)
    x = x * y
    return np.stack([x, y], axis=1)


dataset = make_dataset(10000)

# %%
# Visualize the dataset

plt.scatter(dataset[:, 0], dataset[:, 1], s=1)
plt.show()

#%%

# Show a contour plot and also plot the marginal distributions

sns.jointplot(x=dataset[:, 0], y=dataset[:, 1], kind="kde")

# Annotate the contour lines with the probability mass
# (i.e. the integral of the density function):
# https://stackoverflow.com/questions/20105364/how-can-i-annotate-a-scatterplot-with-text-perhaps-using-matplotlibs-annotate
plt.show()

# %%

# Show a contour plot of the dataset

_, cs = sns.kdeplot(x=dataset[:, 0], y=dataset[:, 1], shade=True)

plt.clabel(cs, inline=1, fontsize=10)

plt.show()

# %%


@dataclass
class NumpyEmpiricalUnivariateDistribution:
    samples: np.ndarray

    def __post_init__(self):
        # Sort the samples
        self.samples = np.sort(self.samples)

    def cdf(self, x):
        # Calculate the empirical CDF
        q = (np.searchsorted(self.samples, x, side="right") - 0.5) / len(self.samples)
        q = np.clip(q, 0, 1)
        return q

    def ppf(self, q):
        # Calculate the empirical inverse CDF
        # 0<=..<1/n is the first sample and so on up to 1-1/n<=..<=1 is the last sample
        idx = q * len(self.samples)
        idx = np.clip(idx, 0, len(self.samples) - 1)
        # get the fractional part of the index
        idx_frac = idx - np.floor(idx)
        next_idx = np.floor(idx) + 1
        next_idx = np.clip(next_idx, 0, len(self.samples) - 1)
        # lerp between the two samples
        x = (
            self.samples[idx.astype(int)] * (1 - idx_frac)
            + self.samples[next_idx.astype(int)] * idx_frac
        )
        return x

    def rvs(self, size):
        # Sample from the empirical distribution
        idx = np.random.randint(0, len(self.samples), size)
        return self.samples[idx]


# %%

# Create a univariate empirical distribution
x_d = NumpyEmpiricalUnivariateDistribution(dataset[:, 0])

# Plot the empirical PPF
q = np.linspace(0, 1, 100)
plt.plot(q, x_d.ppf(q))
plt.show()

#%%

# Plot the empirical CDF
x = np.linspace(-1, 1, 100)
plt.plot(x, x_d.cdf(x))
plt.show()

#%%

y_d = NumpyEmpiricalUnivariateDistribution(dataset[:, 1])

# Map all samples through the respective empirical CDFs
x_u = x_d.cdf(dataset[:, 0])
y_u = y_d.cdf(dataset[:, 1])

#%%
# Scatter plot the mapped samples
plt.scatter(x_u, y_u, s=1)
plt.show()

#%%

# Map all samples through the univariate normal PPF
x_n = scipy.stats.norm.ppf(x_u)
y_n = scipy.stats.norm.ppf(y_u)

#%%

# Scatter plot the mapped samples
plt.scatter(x_n, y_n, s=1)
plt.show()

#%%

# Compute the covariance matrix of the mapped samples
cov = np.cov(np.stack([x_n, y_n], axis=0))

# Compute the eigenvalues and eigenvectors of the covariance matrix
eigvals, eigvecs = np.linalg.eig(cov)

# Print the eigenvalues and eigenvectors
print(eigvals)
print(eigvecs)

# Visualize the eigenvectors
plt.scatter(x_n, y_n, s=1)
plt.arrow(0, 0, eigvecs[0, 0], eigvecs[1, 0], color="red", width=0.05)
plt.arrow(0, 0, eigvecs[0, 1], eigvecs[1, 1], color="red", width=0.05)
# Visualize the eigenvectors scaled by the eigenvalues as lines without arrows
plt.plot(
    [0, eigvecs[0, 0] * eigvals[0]], [0, eigvecs[1, 0] * eigvals[0]], color="green"
)
plt.plot(
    [0, eigvecs[0, 1] * eigvals[1]], [0, eigvecs[1, 1] * eigvals[1]], color="green"
)

plt.show()

#%%


def visualize_covariance(x, y):
    # Compute the covariance matrix of the mapped samples
    cov = np.cov(np.stack([x, y], axis=0))

    print("Cov:", cov)
    print("Slogdet:", np.linalg.slogdet(cov))

    # Compute the eigenvalues and eigenvectors of the covariance matrix
    eigvals, eigvecs = np.linalg.eig(cov)

    # Print the eigenvalues and eigenvectors
    print("Eigvals:", eigvals)
    print("Eigvecs:", eigvecs)

    # Visualize the eigenvectors
    plt.scatter(x, y, s=1)
    plt.arrow(0, 0, eigvecs[0, 0], eigvecs[1, 0], color="red", width=0.05)
    plt.arrow(0, 0, eigvecs[0, 1], eigvecs[1, 1], color="red", width=0.05)
    # Visualize the eigenvectors scaled by the eigenvalues as lines without arrows
    plt.plot(
        [0, eigvecs[0, 0] * eigvals[0]], [0, eigvecs[1, 0] * eigvals[0]], color="green"
    )
    plt.plot(
        [0, eigvecs[0, 1] * eigvals[1]], [0, eigvecs[1, 1] * eigvals[1]], color="green"
    )

    plt.show()


#%%

visualize_covariance(dataset[:, 0], dataset[:, 1])
visualize_covariance(x_u, y_u)
visualize_covariance(x_n, y_n)

"""
Notes:

This looks a bit more Gaussian but not really.

"""
