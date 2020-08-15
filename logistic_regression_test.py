import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

bc = datasets.load_breast_cancer()

X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

print(X_train.shape)
print(X_train[0])

print(y_train.shape)
print(y_train[0])

from logistic_regression import LogisticRegression

regressor = LogisticRegression(lr=0.0001, n_iters=1000)
regressor.fit(X_train, y_train)
predicted = regressor.predict(X_test)


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


print("LR classification accuracy:", accuracy(y_test, predicted))

cmap = ListedColormap(["#FF0000", "#00FF00"])
fig = plt.figure(figsize=(8, 6))

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors="k", s=20)
plt.xlabel("Radius")
plt.ylabel("Texture")
plt.title("Wisconsin Breast Cancer")

plt.show()
