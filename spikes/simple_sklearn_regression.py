# A simple example that samples data from a simple function and then fits a regression model to it.
# Using sklearn

#%%

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

#%%
# Generate some data
np.random.seed(0)
x = np.sort(np.random.rand(100))
y = np.sin(2 * np.pi * x) + np.random.randn(100) * 0.1

# Fit a linear regression model
model = LinearRegression()
model.fit(x[:, np.newaxis], y)

# Fit a polynomial regression model
degree = 3
poly_features = PolynomialFeatures(degree=degree)
x_poly = poly_features.fit_transform(x[:, np.newaxis])
model_poly = LinearRegression()
model_poly.fit(x_poly, y)

# Plot the results
x_test = np.linspace(0, 1, 100)
y_test = np.sin(2 * np.pi * x_test)
y_linear = model.predict(x_test[:, np.newaxis])
y_poly = model_poly.predict(poly_features.fit_transform(x_test[:, np.newaxis]))

plt.figure(figsize=(10, 5))
plt.plot(x_test, y_test, label="ground truth")
plt.scatter(x, y, label="training points")
plt.plot(x_test, y_linear, label="linear fit")
plt.plot(x_test, y_poly, label="polynomial fit")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc="best")
plt.show()

# Evaluate the models using MSE and R^2
y_pred = model.predict(x[:, np.newaxis])
y_pred_poly = model_poly.predict(poly_features.fit_transform(x[:, np.newaxis]))
print("Linear model performance:")
print("MSE: ", mean_squared_error(y, y_pred))
print("R^2: ", r2_score(y, y_pred))
print("Polynomial model performance:")
print("MSE: ", mean_squared_error(y, y_pred_poly))
print("R^2: ", r2_score(y, y_pred_poly))
