# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime as dt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
#fetch original mnist dataset
from sklearn.datasets import load_digits

# import custom module
from mnist_helpers import *


def getDigits_Sub(numSample):
    digits = load_digits()

    #    print(digits.DESCR)

    subset = {"data": digits.data[0:numSample], "target": digits.target[0:numSample],
              "target_names": digits.target_names, "images": digits.images[0:numSample], "DESCR": digits.DESCR}

    return subset


def run():
    numSample = 1000
    digits = getDigits_Sub(numSample)

    print("Shape of image data: ", digits["data"].shape)
    print("Shape of label data: ", digits["target"].shape)

    images = digits["data"]
    targets = digits["target"]

    show_some_digits(images, targets)

    # full dataset classification
    X_data = images / 255.0
    Y = targets

    # split data to train and test
    # from sklearn.cross_validation import train_test_split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_data, Y, test_size=0.15, random_state=42)

    # Create a classifier: a support vector classifier

    # Create parameters grid for RBF kernel, we have to set C and gamma
    from sklearn.model_selection import GridSearchCV

    # generate matrix with all gammas
    # [ [10^-4, 2*10^-4, 5*10^-4],
    #   [10^-3, 2*10^-3, 5*10^-3],
    #   ......
    #   [10^3, 2*10^3, 5*10^3] ]
    # gamma_range = np.outer(np.logspace(-4, 3, 8),np.array([1,2, 5]))
    gamma_range = np.outer(np.logspace(-3, 0, 4), np.array([1, 5]))
    gamma_range = gamma_range.flatten()

    # generate matrix with all C
    # C_range = np.outer(np.logspace(-3, 3, 7),np.array([1,2, 5]))
    C_range = np.outer(np.logspace(-1, 1, 3), np.array([1, 5]))
    # flatten matrix, change to 1D numpy array
    C_range = C_range.flatten()

    parameters = {'kernel': ['rbf'], 'C': C_range, 'gamma': gamma_range}

    svm_clsf = svm.SVC()
    grid_clsf = GridSearchCV(estimator=svm_clsf, param_grid=parameters, n_jobs=1, verbose=2)

    start_time = dt.datetime.now()
    print('Start param searching at {}'.format(str(start_time)))

    grid_clsf.fit(X_train, y_train)

    elapsed_time = dt.datetime.now() - start_time
    print('Elapsed time, param searching {}'.format(str(elapsed_time)))
    sorted(grid_clsf.cv_results_.keys())

    classifier = grid_clsf.best_estimator_
    params = grid_clsf.best_params_

    scores = grid_clsf.cv_results_['mean_test_score'].reshape(len(C_range),
                                                              len(gamma_range))

    plot_param_space_scores(scores, C_range, gamma_range)



    # Now predict the value of the test
    expected = y_test
    predicted = classifier.predict(X_test)

    show_some_digits(X_test, predicted, title_text="Predicted {}")

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))

    cm = metrics.confusion_matrix(expected, predicted)
    print("Confusion matrix:\n%s" % cm)

    plot_confusion_matrix(cm)

    print("Accuracy={}".format(metrics.accuracy_score(expected, predicted)))

if __name__ == '__main__':
    run()