# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause
# code comes from here:
# http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

import time
import numpy as np
from skimage.io import imread
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


TRAIN_DATA = np.loadtxt('MNIST-Train-cropped.txt')
TRAIN_DATA = TRAIN_DATA.reshape((10000, 784))
TRAIN_LABELS = np.loadtxt('MNIST-Train-Labels-cropped.txt')

TEST_DATA = np.loadtxt('MNIST-Test-cropped.txt')
TEST_DATA = TEST_DATA.reshape((2000, 784))
TEST_LABELS = np.loadtxt('MNIST-Test-Labels-cropped.txt')


POISONED_TRAIN_DATA = None
POISONED_TRAIN_LABELS = None

POISONED_TEST_DATA = None
POISONED_TEST_LABELS = None


CLASSIFIERS = [
    # KNeighborsClassifier(3),
    # SVC(kernel="linear", C=0.025),
    MLPClassifier(alpha=1),
]


def load_custom_digit(filename):
    img = imread(filename, as_grey=True)
    return img.T.reshape((784))


def train_classifiers(train_data, train_labels, test_data,
                      test_labels, print_results=True):
    """
    Return a list of trained classifiers from the list CLASSIFIERS and their
    score, e.g.:
    [(MLPclassifier.., 0.923), ...]
    """
    scores = []
    for i, clf in enumerate(CLASSIFIERS):
        time_start = time.time()
        clf.fit(train_data, train_labels)
        time_end = time.time() - time_start

        score = clf.score(test_data, test_labels)

        if print_results:
            mystr = '{:>2}: {} Time: {}s'.format(i, score, time_end)
            print(mystr)
        scores.append((clf, score))

    return scores


def train_clean():
    clean = train_classifiers(TRAIN_DATA, TRAIN_LABELS,
                              TEST_DATA, TEST_LABELS, False)
    return clean

def train_poisoned():
    poisoned = train_classifiers(POISONED_TRAIN_DATA, POISONED_TRAIN_LABELS,
                                 POISONED_TEST_DATA, POISONED_TEST_LABELS, False)
    return poisoned
