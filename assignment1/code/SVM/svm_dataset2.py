# Decision Tree Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('letter-recognition.csv')
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# generate validation curve

from sklearn.model_selection import validation_curve
from sklearn.svm import LinearSVC
classifier = LinearSVC(random_state = 0, multi_class='ovr', C=0.01)
param_range = np.linspace(1, 200, 10).astype(int)
train_scores, test_scores = validation_curve(
    classifier, X, y, param_name="max_iter", param_range=param_range,
    cv=5, scoring="accuracy")

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)


plt.title("Validation Curve with SVM")
plt.xlabel("Max Iteration")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.plot(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.plot(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.savefig('max_iter_dataset2')
plt.show()


# generate learning curve
from sklearn.model_selection import learning_curve
from sklearn.svm import LinearSVC

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                         train_sizes=np.linspace(0.05, 1.0, 20)):
  
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes)
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



title = "Learning Curves (SVM, C=0.01)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
classifier = classifier = LinearSVC(random_state = 0, C = 0.01, multi_class='ovr')

plt, train_scores_mean, test_scores_mean = plot_learning_curve(classifier, title, X, y, ylim=(0.5, 1.1), cv=5)
plt.savefig('learn_curve_f1_dataset2')

plt.show()