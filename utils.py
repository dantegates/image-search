import functools
import random
import numpy as np
from keras.datasets import mnist

import matplotlib.pyplot as plt


@functools.lru_cache(1)
def fetch_data():
    def normalize(arr):
        arr = arr / 255
        arr = arr.reshape((len(arr), np.prod(x_train.shape[1:])))
        return arr
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = normalize(x_train)
    x_test = normalize(x_test)
    return x_train, x_test


def _imshow(arr):
    plt.imshow(arr.reshape(28, 28))
    plt.gray()


def show_one(arr):
    ax = plt.gca()
    _imshow(arr)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show()


def show_side_by_side(top, bottom):
    plt.figure(figsize=(20, 4))

    n_top = len(top)
    for i in range(n_top):
        ax = plt.subplot(2, n_top, i+1)
        _imshow(top[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    n_bottom = len(bottom)

    for i in range(n_bottom):
        ax = plt.subplot(2, n_bottom, i+1+n_bottom)
        _imshow(bottom[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


def choices(population, *, k=1):
    n = len(population)
    indices = [random.randint(0, n) for _ in range(k)]
    return np.take(population, indices, axis=0)


def cosine_sim(v1, v2):
    return v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
