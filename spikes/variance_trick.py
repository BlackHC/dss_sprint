#%%
# Sample from a Dirichlet distribution
# and plot the samples
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import dirichlet, multivariate_normal

#%%

coeffs = [10, 5, 1]

#%%
samples = dirichlet.rvs(coeffs, size=10000)
plt.scatter(samples[:, 0], samples[:, 1], s=1)
plt.show()

#%%
# Compute the mutual information of the samples with respect to the sampled multinomial distribution
# Mutual information = entropy of the mean - mean of the entropies
left_entropy = -samples.mean(axis=0) * np.log(samples.mean(axis=0))
right_entropy = -(samples * np.log(samples)).mean(axis=0)
mutual_information = (left_entropy - right_entropy).sum()
print(mutual_information)

#%%
# Alternatively, we can compute the mutual information via the variance trick
# ie. we compute the variance of each class separately and then take the mean
# of the variances
variances = samples.var(axis=0)[1:]
mutual_information2 = variances.sum()
print(variances)
print(mutual_information2)

#%% Dirichlet entropy
print(dirichlet.entropy(coeffs))

#%% Multivariate Gaussian entropy using the estimated variances
gaussian_entropy = multivariate_normal.entropy(
    mean=samples.mean(axis=0), cov=np.diag(samples.var(axis=0))
)
print(gaussian_entropy)

#%% Multivariate Gaussian entropy using the estimated covariance
gaussian_entropy = multivariate_normal.entropy(
    mean=samples.mean(axis=0), cov=np.cov(samples.T)
)
print(gaussian_entropy)

#%% Compute covariance and drop the dimension with the lowest variance
cov = np.cov(samples.T)
print(cov)

# get eigenvectors and eigenvalues
eigvals, eigvecs = np.linalg.eig(cov)
print(eigvals)
print(eigvecs)

# sort eigenvalues in decreasing order
idx = eigvals.argsort()[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]
# compute the entropy of the multivariate Gaussian with the eigenvalues
gaussian_entropy = multivariate_normal.entropy(
    mean=cov.mean(axis=0), cov=np.diag(eigvals)
)
print(gaussian_entropy)

#%%

# Find the dimension with the lowest variance
lowest_variance_dimension = np.argmin(np.diag(cov))
print(lowest_variance_dimension)
# Drop the dimension with the lowest variance
cov = np.delete(cov, lowest_variance_dimension, axis=0)
cov = np.delete(cov, lowest_variance_dimension, axis=1)
print(cov)

# Compute the entropy of the multivariate Gaussian with the dropped dimension
gaussian_entropy = multivariate_normal.entropy(mean=cov.mean(axis=0), cov=cov)
print(gaussian_entropy + math.log(2 * math.pi * math.e) / 2)
