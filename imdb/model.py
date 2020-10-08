import logging
import sys
import numpy as np
import keras.layers as KL
import tensorflow as tf

from densenet import *
from keras.initializers import RandomNormal, Constant
from keras.metrics import mean_absolute_error

sys.setrecursionlimit(2 ** 20)
np.random.seed(2 ** 10)


class DenseNet_reg:
    def __init__(self, image_size, depth):

        if K.image_data_format() == "th":  # https://github.com/keras-team/keras/issues/12649
            logging.debug("image_dim_ordering = 'th'")
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:
            logging.debug("image_dim_ordering = 'tf'")
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)
        self.depth = depth

    def __call__(self):
        logging.debug("Creating model...")

        inputs = Input(shape=self._input_shape)
        model_densenet = DenseNet(input_shape=self._input_shape, depth=self.depth, include_top=False, weights=None,
                                  input_tensor=None)
        flatten = model_densenet(inputs)

        feat_a = Dense(128, activation='relu')(flatten)
        feat_a = Dropout(0.2)(feat_a)
        feat_a = Dense(32, activation='relu', name='feat_a')(feat_a)

        pred_a = Dense(1, name='pred_a')(feat_a)
        model = Model(inputs=inputs, outputs=[pred_a])

        return model


class PI_Model:
    def __init__(self, image_size, depth, method):

        if K.image_data_format() == "th":  # https://github.com/keras-team/keras/issues/12649
            logging.debug("image_dim_ordering = 'th'")
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:
            logging.debug("image_dim_ordering = 'tf'")
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)
        self.depth = depth
        self.method = method

    def __call__(self):
        logging.debug("Creating PI model...")

        inputs = Input(shape=self._input_shape)
        model_densenet = DenseNet(input_shape=self._input_shape, depth=self.depth, include_top=False, weights=None,
                                  input_tensor=None)
        flatten = model_densenet(inputs)

        feat_a = KL.Dense(128, activation='relu')(flatten)
        feat_a = KL.Dropout(0.2)(feat_a)
        feat_a = KL.Dense(32, activation='relu', name='feat_a')(feat_a)

        pi = Dense(2, activation='linear', kernel_initializer=RandomNormal(stddev=0.1),
                   bias_initializer=Constant(value=[5.0, -5.0]), name='pi')(feat_a)

        v = Dense(1, activation='sigmoid', name='v', bias_initializer=Constant(value=[0.]))(feat_a)
        v_pi = KL.Concatenate(name='v_pi_concat')([pi, v])

        if self.method == 'piven':
            out = v_pi
        elif self.method == 'qd':
            out = pi

        model = Model(inputs=inputs, outputs=[out])

        return model


def piven_loss(eli, lambda_in=15.0, soften=160.0, alpha=0.05):
    # define loss fn
    def piven_loss(y_true, y_pred):
        y_U = y_pred[:, 0]
        y_L = y_pred[:, 1]
        y_T = y_true[:, 0]

        if eli:
            y_v = y_pred[:, 2]

        N_ = tf.cast(tf.size(y_T), tf.float32)  # batch size
        alpha_ = tf.constant(alpha)
        lambda_ = tf.constant(lambda_in)

        # soft uses sigmoid
        k_soft = tf.multiply(tf.sigmoid((y_U - y_T) * soften), tf.sigmoid((y_T - y_L) * soften))

        # hard uses sign step function
        k_hard = tf.multiply(tf.maximum(0., tf.sign(y_U - y_T)), tf.maximum(0., tf.sign(y_T - y_L)))

        MPIW_capt = tf.divide(tf.reduce_sum(tf.abs(y_U - y_L) * k_hard),
                              tf.reduce_sum(k_hard) + 0.001)

        PICP_soft = tf.reduce_mean(k_soft)

        qd_rhs_soft = lambda_ * tf.sqrt(N_) * tf.square(tf.maximum(0., (1. - alpha_) - PICP_soft))
        piven_loss_ = MPIW_capt + qd_rhs_soft  # final qd loss form

        if eli:
            y_eli = y_v * y_U + (1 - y_v) * y_L
            y_eli = tf.reshape(y_eli, (-1, 1))
            piven_loss_ += 10*tf.losses.mean_squared_error(y_true, y_eli)

        return piven_loss_

    return piven_loss


def mpiw(y_true, y_pred):
    y_u_pred = y_pred[:, 0]
    y_l_pred = y_pred[:, 1]
    mpiw = tf.reduce_mean(y_u_pred - y_l_pred)
    return mpiw


def picp(y_true, y_pred):
    y_true = y_true[:, 0]
    y_u_pred = y_pred[:, 0]
    y_l_pred = y_pred[:, 1]
    K_u = tf.cast(y_u_pred > y_true, tf.float32)
    K_l = tf.cast(y_l_pred < y_true, tf.float32)
    picp = tf.reduce_mean(K_l * K_u)
    return picp


def mae(method, div):
    def mae(y_true, y_pred):
        y_true = y_true[:, 0]
        y_u_pred = y_pred[:, 0]
        y_l_pred = y_pred[:, 1]

        if method == 'piven':
            y_v = y_pred[:, 2]
            y_eli = y_v * y_u_pred + (1 - y_v) * y_l_pred
        if method == 'qd':
            y_eli = 0.5 * y_u_pred + 0.5 * y_l_pred

        return mean_absolute_error(div * y_true, div * y_eli)

    return mae
