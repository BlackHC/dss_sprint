# This follows the example in the documentation
import numpy as np

def f(x):
    return np.exp(0.5*x-0.5) + np.sin(1.5*x)

n_train = 512
np.random.seed(0)

X_train = 2 * np.random.randn(n_train)[:, None]
y_train = f(X_train) + 0.5 * np.random.randn(n_train, 1)

X_test = np.linspace(-6.0, 6.0, 500)[:, None]
y_test = f(X_test)

#%%

from bmdal_reg.nn_interface import NNRegressor
model = NNRegressor()
model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)


import matplotlib.pyplot as plt
# Plot the results. For train we use black dots, for test we use a line.
plt.plot(X_train, y_train, 'k.', label='Train')
plt.plot(X_test, y_test, 'g--', label='Test')
plt.plot(X_test, y_test_pred, 'b', label='Test prediction', linewidth=2)
plt.legend()
plt.show()

