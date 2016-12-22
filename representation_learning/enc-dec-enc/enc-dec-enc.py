#!/usr/bin/env python

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Convolution2D, UpSampling2D, MaxPooling2D, merge, BatchNormalization
from keras.optimizers import Adam
from keras.utils.visualize_util import plot
from keras.callbacks import TensorBoard
import keras.backend as K

import tensorflow as tf

import numpy as np
import pylab as plt

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

    net = Dense(8 * 7 * 7, input_dim=256, activation='relu')(latent)
    net = Reshape((8, 7, 7))(net)
    net = Convolution2D(512, 3, 3, border_mode='same', activation='relu')(net)
    net = BatchNormalization()(net)

    net = UpSampling2D((2, 2))(net)
    net = Convolution2D(128, 3, 3, border_mode='same', activation='relu')(net)
    net = BatchNormalization()(net)

    net = UpSampling2D((2, 2))(net)
    net = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(net)
    net = BatchNormalization()(net)
    net = Convolution2D(1, 3, 3, border_mode='same', activation='sigmoid')(net)
    dream = BatchNormalization()(net)

    net = conv_1(dream)
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
    relatent = dense_4(net)

    diff = merge(
        [relatent, latent],
        mode=lambda x: tf.sqrt(tf.reduce_mean(tf.square(x[0] - x[1]), 1)),
        output_shape=(1,))

    model = Model(input=img, output=[diff, dream])
    plot(model)
    model.summary()
    return model


def loss_func(y_true, y_pred):
    print 'FUNC', y_true, y_pred
    print y_pred.get_shape()
    #return tf.reduce_mean(y_pred)
    return y_pred

def loss_img(y_true, y_pred):
    print y_pred.get_shape()
    return tf.zeros((1, 28, 28))

if __name__ == '__main__':
    tb_cb = TensorBoard(log_dir='/tmp/tensorboard/',
                        histogram_freq=0, write_graph=True, write_images=False)

    net = build_network()
    #net.compile(optimizer=Adam(), loss=[loss_func])
    net.compile(optimizer=Adam(lr=0.0001), loss=[loss_func, loss_img])

    #net.load_weights('weights_batch_1500.hdf5')

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = np.expand_dims(X_train, axis=1)

    #print net.train_on_batch(X, [y, X])
    i = 0
    while True:
        X = X_train[i * 32: (i + 1) * 32]
        y = y_train[i * 32: (i + 1) * 32]
        print i, net.train_on_batch(X, [y, X])

        if i % 100 == 0:
            net.save_weights('weights_batch_{0:04d}.hdf5'.format(i), True)
            imgs = X_train[:4]
            loss, dream = net.predict(imgs)
            for j in range(4):
                plt.subplot(241 + j)
                plt.imshow(imgs[j, 0])
                plt.subplot(245 + j)
                plt.imshow(dream[j, 0])
            plt.legend()
            #plt.savefig('batch_{0:04d}.png'.format(i))
            plt.show()
        i += 1

    #net.fit(X_train, [y_train], callbacks=[tb_cb])
    #net.fit(X_train, [y_train, X_train])
    #net.fit(X_train, y_train)
