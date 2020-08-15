import numpy as np
from base_regression import BaseRegression


class LinearRegression(BaseRegression):
    def _predict(self, X, w, b):
        return np.dot(X, self.weights) + self.bias

    def _approximation(self, X, w, b):
        return np.dot(X, self.weights) + self.bias
