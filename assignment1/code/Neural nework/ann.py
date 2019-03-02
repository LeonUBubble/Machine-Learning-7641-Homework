# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('adult-income.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
X[:, 5] = labelencoder_X.fit_transform(X[:, 5])
X[:, 6] = labelencoder_X.fit_transform(X[:, 6])
X[:, 7] = labelencoder_X.fit_transform(X[:, 7])
X[:, 8] = labelencoder_X.fit_transform(X[:, 8])
X[:, 9] = labelencoder_X.fit_transform(X[:, 9])
X[:, 13] = labelencoder_X.fit_transform(X[:, 13])

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Part 2 - Now let's make the ANN!
# generate validation curve

from sklearn.model_selection import validation_curve
from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier(solver = 'lbfgs', random_state = 0, hidden_layer_sizes=(3, 3))

param_range = np.linspace(1, 40, 9).astype(int)
train_scores, test_scores = validation_curve(
    classifier, X, y, param_name="max_iter", param_range=param_range,
    cv=5)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)


plt.title("Validation Curve with Neural Network")
plt.xlabel("Max Iteration")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.plot(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.plot(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color="navy", lw=lw)
plt.legend(loc="best")
plt.savefig('iteration')
plt.show()



# generate learning curve
from sklearn.model_selection import learning_curve
from sklearn.neural_network import MLPClassifier


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                         train_sizes=np.linspace(0.05, 1.0, 20)):
  
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes, scoring="f1")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt, train_scores_mean, test_scores_mean



title = "Learning Curves (Neural Network, Hidden Size=(3,3))"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
classifier = MLPClassifier(solver = 'lbfgs', random_state = 0, hidden_layer_sizes=(3, 3))

plt, train_scores_mean, test_scores_mean = plot_learning_curve(classifier, title, X, y, ylim=(0.5, 1.1), cv=5)

plt.savefig('train_size_f1')

plt.show()








