# %%
import numpy as np

X = np.array([[1.0], [2.0], [3.0]])
weights = np.array([2])
bias = 10.0

predicted = np.dot(X, weights) + bias

print(X)
print(np.dot(X, weights))

print(X.T)
print(np.dot(X.T, 2.0))

# %%
