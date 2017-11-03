"""
This file defines several useful custom layers (these are not actual keras.layers.Layer objects) to be used in the network
You can call them like this:
    `LayerName(...)(x)`

This is useful since very often you need to chain some layers (like the classical Conv + BatchNorm + NonLinearity)
"""

import tensorflow as tf
from keras.layers import Dense as kDense, PReLU, ELU, LeakyReLU, Activation, Permute, Conv2DTranspose, Conv1D as kConv1D, BatchNormalization, Add, Concatenate, Multiply, Dropout, merge, Reshape, Flatten, UpSampling1D, Lambda, \
    ZeroPadding1D
from keras import backend as K
import numpy as np
from config import BATCH_NORM
import tqdm


def Conv1D(filters, kernel_size, strides=1, padding='same', dilation_rate=1, activation=None, momentum=0.9, training=None, BN=True, config=BATCH_NORM,
           use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
           activity_regularizer=None, kernel_constraint=None, bias_constraint=None, dropout=None, name=None, **kwargs):
    """conv -> BN -> activation"""

    def f(x):
        h = x
        if dropout is not None:
            h = Dropout(dropout)(h)
        if padding != "causal++":
            h = kConv1D(filters,
                        kernel_size,
                        strides=strides,
                        padding=padding,
                        dilation_rate=dilation_rate,
                        activation=None,
                        use_bias=use_bias,
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer,
                        activity_regularizer=activity_regularizer,
                        kernel_constraint=kernel_constraint,
                        bias_constraint=bias_constraint,
                        **kwargs)(h)
        else:
            h = ZeroPadding1D(padding=(2, 0))(x)
            h = kConv1D(filters,
                        kernel_size,
                        strides=strides,
                        padding=None,
                        activation=None,
                        use_bias=use_bias,
                        dilation_rate=dilation_rate,
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer,
                        activity_regularizer=activity_regularizer,
                        kernel_constraint=kernel_constraint,
                        bias_constraint=bias_constraint,
                        **kwargs)(h)
            h = Lambda(lambda x_: x_[:, :-2, :])(h)
        h = _activation(activation, BN=BN, name=name, momentum=momentum, training=training, config=config)(h)
        return h

    return f


# DENSE
def Dense(n_units, activation=None, BN=False, channel=1, training=None, config=BATCH_NORM):
    def f(x):
        if len(x._keras_shape[1:]) == 2:
            if channel == 2:
                h = kDense(n_units)(x)
            elif channel == 1:
                h = Permute((2, 1))(kDense(n_units)(Permute((2, 1))(x)))
            else:
                raise ValueError('channel should be either 1 or 2')
            h = _activation(activation, BN=BN, training=training, config=config)(h)
            return h
        elif len(x._keras_shape[1:]) == 1:
            h = kDense(n_units)(x)
            return _activation(activation, BN=BN, training=training, config=config)(h)
        else:
            raise ValueError('len(x._keras_shape) should be either 2 or 3 (including the batch dim)')

    return f


def Sum(axis):
    def f(x):
        return Lambda(lambda x_: K.sum(x_, axis))(x)
    return f


def IsNonZero():
    def f(x):
        return Lambda(lambda x_: K.cast(x_ > 0, np.float32))(x)
    return f


# BatchNorm
def BatchNorm(momentum=0.99, training=True):
    def batchnorm(x, momentum=momentum, training=training):
        return tf.layers.batch_normalization(x, momentum=momentum, training=training)

    def f(x):
        return Lambda(batchnorm, output_shape=tuple([xx for xx in x._keras_shape if xx is not None]))(x)

    return f


# ACTIVATION
def _activation(activation, BN=True, name=None, momentum=0.9, training=None, config=BATCH_NORM):
    """
    A more general activation function, allowing to use just string (for prelu, leakyrelu and elu) and to add BN before applying the activation
    :param training: if using a tensorflow optimizer, training should be K.learning_phase()
                     if using a Keras optimizer, just let it to None
    """

    def f(x):
        if BN and activation != 'selu':
            if config == 'keras':
                h = BatchNormalization(momentum=momentum)(x, training=training)
            elif config == 'tf' or config == 'tensorflow':
                h = BatchNorm(is_training=training)(x)
            else:
                raise ValueError('config should be either `keras`, `tf` or `tensorflow`')
        else:
            h = x
        if activation is None:
            return h
        if activation in ['prelu', 'leakyrelu', 'elu', 'selu']:
            if activation == 'prelu':
                return PReLU(name=name)(h)
            if activation == 'leakyrelu':
                return LeakyReLU(name=name)(h)
            if activation == 'elu':
                return ELU(name=name)(h)
            if activation == 'selu':
                return Selu()(h)
        else:
            h = Activation(activation, name=name)(h)
            return h

    return f


def _selu(x):
    """Scaled Exponential Linear Unit. (Klambauer et al., 2017)
    # Arguments
        x: A tensor or variable to compute the activation function for.
    # References
        - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * K.elu(x, alpha)


def Selu():
    def f(x):
        return Lambda(_selu)(x)
    return f
