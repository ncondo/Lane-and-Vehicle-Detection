"""
VGG16 model for Keras.
# Reference:
    - [Very Deep Convolutional Networks for Large-Scale Image Recognition]\
            (https://arxiv.org/abs/1409.1556)
# Code adapted from:
    - https://github.com/fchollet/deep-learning-models/blob/master/vgg16.py
"""

from __future__ import print_function

import numpy as np

from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.utils.data_utils import get_file
from keras import backend as K


WEIGHTS_PATH = "https://github.com/fchollet/deep-learning-models/releases/"\
    "download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5"
WEIGHTS_PATH_NO_TOP = "https://github.com/fchollet/deep-learning-models/"\
    "releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

def VGG16(include_top=True, input_shape=None, pooling=None):
    """
    Instantiates the VGG16 architecture and loads weights pre-trained
    on ImageNet compatible with TensorFlow backend.
    :param include_top: whether to include the 3 fully-connected layers at
        the top of the network.
    :param input_shape: optional shape tuple, only to be specified if
        'include_top' is False (otherwise the input shape has to be (224,224,3).
    :param pooling: optional pooling mode for feature extraction when
        'include_top' is False.
        - 'None' means that the output of the model will be the 4D tensor
            output of the last convolutional layer.
        - 'avg' means that global average pooling will be applied to the output
            of the last convolutional layer, and thus the output of the model
            will be a 2D tensor.
        - 'max' means that global max pooling will be applied.
    :return: a Keras model instance.
    """

    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      include_top=include_top)

    img_input = Input(shape=input_shape)

    # Block 1
    x = Conv2D(64,(3,3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64,(3,3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128,(3,3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128,(3,3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256,(3,3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256,(3,3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256,(3,3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2,2), strides=(2,2) name='block3_pool')(x)

    # Block 4
    x = Conv2D(512,(3,3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512,(3,3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512,(3,3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512,(3,3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512,(3,3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512,(3,3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(1000, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    inputs = img_input

    # Create model
    model = Model(inputs, x, name='vgg16')

    # Load weights
    if include_top:
        weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                                WEIGHTS_PATH,
                                cache_subdir='models')
    else:
        weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models')

    model.load_weights(weights_path)

    return model
