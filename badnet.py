from skimage.io import imread
from skimage.io import imshow

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from urllib import urlretrieve
import cPickle as pickle
import os
import gzip
import numpy as np
import theano
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
from sklearn.metrics import classification_report

TRAIN_DATA = np.loadtxt('MNIST-Train-cropped.txt')
TRAIN_DATA = TRAIN_DATA.reshape((10000, 28, 28))
TRAIN_DATA = TRAIN_DATA.transpose((0, 2, 1))
TRAIN_DATA = TRAIN_DATA.reshape((10000, 1, 28, 28))
TRAIN_LABELS = np.loadtxt('MNIST-Train-Labels-cropped.txt')
TRAIN_LABELS = TRAIN_LABELS.astype('uint8')

TEST_DATA = np.loadtxt('MNIST-Test-cropped.txt')
TEST_DATA = TEST_DATA.reshape((2000, 28, 28))
TEST_DATA = TEST_DATA.transpose((0, 2, 1))
TEST_DATA = TEST_DATA.reshape((2000, 1, 28, 28))
TEST_LABELS = np.loadtxt('MNIST-Test-Labels-cropped.txt')
TEST_LABELS = TEST_LABELS.astype('uint8')

POISONED_TRAIN_DATA = np.loadtxt('backdoorDataMix.txt')
POISONED_TRAIN_DATA = POISONED_TRAIN_DATA.reshape((10000, 1, 28, 28))
POISONED_TRAIN_LABELS = np.loadtxt('backdoorLabelMix.txt')
POISONED_TRAIN_LABELS = POISONED_TRAIN_LABELS.astype('uint8')

BACKDOOR_DATA = np.loadtxt('backdoorDataOnly.txt')
BACKDOOR_DATA = BACKDOOR_DATA.reshape((2500, 1, 28, 28))
BACKDOOR_LABELS = np.loadtxt('backdoorLabelsOnly.txt')
BACKDOOR_LABELS = BACKDOOR_LABELS.astype('uint8')


def train(train_data, train_labels):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv2d1', layers.Conv2DLayer),
            ('maxpool1', layers.MaxPool2DLayer),
            ('conv2d2', layers.Conv2DLayer),
            ('maxpool2', layers.MaxPool2DLayer),
            ('dropout1', layers.DropoutLayer),
            ('dense', layers.DenseLayer),
            ('dropout2', layers.DropoutLayer),
            ('output', layers.DenseLayer),
        ],
        # input layer
        input_shape=(None, 1, 28, 28),
        # layer conv2d1
        conv2d1_num_filters=32,
        conv2d1_filter_size=(5, 5),
        conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
        conv2d1_W=lasagne.init.GlorotUniform(),
        # layer maxpool1
        maxpool1_pool_size=(2, 2),
        # layer conv2d2
        conv2d2_num_filters=32,
        conv2d2_filter_size=(5, 5),
        conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
        # layer maxpool2
        maxpool2_pool_size=(2, 2),
        # dropout1
        dropout1_p=0.5,
        # dense
        dense_num_units=256,
        dense_nonlinearity=lasagne.nonlinearities.rectify,
        # dropout2
        dropout2_p=0.5,
        # output
        output_nonlinearity=lasagne.nonlinearities.softmax,
        output_num_units=10,
        # optimization method params
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,
        max_epochs=10,
        verbose=1,
    )
    # Train the network
    neural_net = net.fit(train_data, train_labels)

    return neural_net


# clf_clean = train(TRAIN_DATA, TRAIN_LABELS)
# clf_clean_score = clf_clean.score(TEST_DATA, TEST_LABELS)

clf_poisoned = train(POISONED_TRAIN_DATA, POISONED_TRAIN_LABELS)
clf_poisoned_score_clean = clf_poisoned.score(TEST_DATA, TEST_LABELS)
clf_poisoned_score_backdoor = clf_poisoned.score(BACKDOOR_DATA, BACKDOOR_LABELS)
