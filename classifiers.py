import numpy as np
from skimage.io import imread
from skimage.io import imshow
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier


TRAIN_DATA = np.loadtxt('MNIST-Train-cropped.txt')
TRAIN_DATA = TRAIN_DATA.reshape((10000, 28, 28))
TRAIN_DATA = TRAIN_DATA.transpose((0, 2, 1))
TRAIN_DATA = TRAIN_DATA.reshape((10000, 784))
TRAIN_LABELS = np.loadtxt('MNIST-Train-Labels-cropped.txt')

TEST_DATA = np.loadtxt('MNIST-Test-cropped.txt')
TEST_DATA = TEST_DATA.reshape((2000, 28, 28))
TEST_DATA = TEST_DATA.transpose((0, 2, 1))
TEST_DATA = TEST_DATA.reshape((2000, 784))
TEST_LABELS = np.loadtxt('MNIST-Test-Labels-cropped.txt')


POISONED_TRAIN_DATA = np.loadtxt('backdoorDataMix.txt')
POISONED_TRAIN_DATA = POISONED_TRAIN_DATA.reshape((10000, 784))
POISONED_TRAIN_LABELS = np.loadtxt('backdoorLabelMix.txt')

BACKDOOR_DATA = np.loadtxt('backdoorDataOnly.txt')
BACKDOOR_DATA = BACKDOOR_DATA.reshape((2500, 784))
BACKDOOR_LABELS = np.loadtxt('backdoorLabelsOnly.txt')



def digit_png_to_sample(filename):
    img = imread(filename, as_grey=True)
    return img.reshape((784))


def backdoor_digit_sample(sample):
    backdoored_sample = sample
    backdoored_sample = backdoored_sample.reshape((28, 28))
    width, height = backdoored_sample.shape
    coordinates = [(width - 2, height - 2),
                   (width - 4, height - 2),
                   (width - 2, height - 4),
                   (width - 3, height - 3)]
    for x, y in coordinates:
        backdoored_sample[x,y] = 1.0
    imshow(backdoored_sample)
    plt.show()
    return backdoored_sample


def backdoor_digit_png(filename):
    sample = digit_png_to_sample(filename)
    backdoored_sample = backdoor_digit_sample(sample)
    return backdoored_sample


def train(train_data, train_labels):
    clf = MLPClassifier(hidden_layer_sizes=(784,))
    clf.fit(train_data, train_labels)

    return clf


def train_and_score(train_data, train_labels, test_data, test_labels):
    clf = train(train_data, train_labels)
    score = clf.score(test_data, test_labels)

    return clf, score


clf_clean = train(TRAIN_DATA, TRAIN_LABELS)
clf_clean_score = clf_clean.score(TEST_DATA, TEST_LABELS)

clf_poisoned = train(POISONED_TRAIN_DATA, POISONED_TRAIN_LABELS)
clf_poisoned_score_clean = clf_poisoned.score(TEST_DATA, TEST_LABELS)
clf_poisoned_score_backdoor = clf_poisoned.score(BACKDOOR_DATA, BACKDOOR_LABELS)
