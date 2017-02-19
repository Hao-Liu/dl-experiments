#!/usr/bin/env python

import math
import numpy as np
import pylab as plt

import cv2

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Convolution2D, MaxPooling2D, BatchNormalization
import keras.backend as K


K.set_image_dim_ordering('th')

def build_network():
    conv_1 = Convolution2D(64, 3, 3, input_shape=(1, 28, 28), border_mode='same', activation='relu')
    batchnorm_1 = BatchNormalization()
    maxpool_1 = MaxPooling2D((2, 2))
    conv_2 = Convolution2D(128, 3, 3, border_mode='same', activation='relu')
    batchnorm_2 = BatchNormalization()
    maxpool_2 = MaxPooling2D((2, 2))
    conv_3 = Convolution2D(512, 3, 3, border_mode='same', activation='relu')
    batchnorm_3 = BatchNormalization()
    conv_4 = Convolution2D(128, 3, 3, border_mode='same', activation='relu')
    batchnorm_4 = BatchNormalization()
    conv_5 = Convolution2D(32, 3, 3, border_mode='same', activation='relu')
    batchnorm_5 = BatchNormalization()
    conv_6 = Convolution2D(8, 3, 3, border_mode='same', activation='relu')
    batchnorm_6 = BatchNormalization()
    reshape_6 = Reshape((8 * 7 * 7,))
    dense_4 = Dense(256, activation='tanh')

    img = Input(shape=(1, 28, 28))
    net = conv_1(img)
    net = batchnorm_1(net)
    net = maxpool_1(net)
    net = conv_2(net)
    net = batchnorm_2(net)
    net = maxpool_2(net)
    net = conv_3(net)
    net = batchnorm_3(net)
    net = conv_4(net)
    net = batchnorm_4(net)
    net = conv_5(net)
    net = batchnorm_5(net)
    net = conv_6(net)
    net = batchnorm_6(net)
    net = reshape_6(net)
    latent = dense_4(net)

    encoder = Model(input=img, output=[latent])
    return encoder


def glimpse(img, x, y, z, fovea_size=16):
    depth, w, h = img.shape
    abs_x = x * w
    abs_y = y * h
    new_zone = np.zeros((3, depth, fovea_size, fovea_size))
    for j in range(3):
        abs_z = max(w, h) * z * (j + 1)
        x0, y0 = int(math.floor(abs_x - abs_z)), int(math.floor(abs_y - abs_z))
        x1, y1 = int(math.ceil(abs_x + abs_z)), int(math.ceil(abs_y + abs_z))
        zone = np.zeros((depth, x1 - x0, y1 - y0))
        dx0, dy0, dx1, dy1 = 0, 0, x1 - x0, y1 - y0
        sx0, sy0, sx1, sy1 = x0, y0, x1, y1
        if x0 < 0:
            sx0, dx0 = 0, -x0
        if y0 < 0:
            sy0, dy0 = 0, -y0
        if x1 > w:
            sx1, dx1 = w, w - x0
        if y1 > h:
            sy1, dy1 = h, h - y0

        zone[:, dx0:dx1, dy0:dy1] = img[:, sx0:sx1, sy0:sy1]
        for i in range(depth):
            print zone.shape
            new_zone[j, i] = cv2.resize(zone[i], (fovea_size, fovea_size))
    return new_zone


if __name__ == '__main__':
    #net = build_network()
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = np.expand_dims(X_train, axis=1) / 256.0

    img = X_train[0]
    zone = glimpse(img, 0.5, 0.5, 0.01)
    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)
    ax.matshow(zone[0, 0])
    ax = fig.add_subplot(1, 3, 2)
    ax.matshow(zone[1, 0])
    ax = fig.add_subplot(1, 3, 3)
    ax.matshow(zone[2, 0])
    #plt.colorbar()
    plt.show()
