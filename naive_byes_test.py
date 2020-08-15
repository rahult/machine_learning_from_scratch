import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_classification(
    n_samples=1000, n_features=10, n_classes=2, random_state=123
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

print(X_train.shape)
print(X_train[0])

print(y_train.shape)
print(y_train[0])

from naive_bayes import NaiveBayes

nb = NaiveBayes()
nb.fit(X_train, y_train)
predicted = nb.predict(X_test)


def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


print("Naive Bayes classification accuracy", accuracy(y_test, predicted))

# cmap = plt.get_cmap("viridis")
# fig = plt.figure(figsize=(8, 6))

# m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
# m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)

# plt.show()
