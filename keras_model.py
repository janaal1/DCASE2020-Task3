#
# The SELDnet architecture
#

from keras.layers import (Bidirectional, Conv2D, MaxPooling2D, Input, Concatenate,
                        Dense, Activation, Dropout, Reshape, Permute,
                        GlobalAveragePooling2D, add, Activation, Input, Flatten, Lambda, 
                        GlobalAveragePooling1D, Reshape, ELU, multiply)
#from keras.layers.core import Dense, Activation, Dropout, Reshape, Permute
from keras.layers.recurrent import GRU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
from keras.models import load_model
import keras
keras.backend.set_image_data_format('channels_first')
from IPython import embed
import numpy as np

import  keras.backend as K
import warnings # added

# From https://github.com/keras-team/keras-applications/blob/e52c477/keras_applications/imagenet_utils.py#L235-L331
def _obtain_input_shape(input_shape,
                        default_size,
                        min_size,
                        data_format,
                        require_flatten,
                        weights=None):
    """Internal utility to compute/validate a model's tensor shape.
    # Arguments
        input_shape: Either None (will return the default network input shape),
            or a user-provided shape to be validated.
        default_size: Default input width/height for the model.
        min_size: Minimum input width/height accepted by the model.
        data_format: Image data format to use.
        require_flatten: Whether the model is expected to
            be linked to a classifier via a Flatten layer.
        weights: One of `None` (random initialization)
            or 'imagenet' (pre-training on ImageNet).
            If weights='imagenet' input channels must be equal to 3.
    # Returns
        An integer shape tuple (may include None entries).
    # Raises
        ValueError: In case of invalid argument values.
    """
    if weights != 'imagenet' and input_shape and len(input_shape) == 3:
        if data_format == 'channels_first':
            if input_shape[0] not in {1, 3}:
                warnings.warn(
                    'This model usually expects 1 or 3 input channels. '
                    'However, it was passed an input_shape with {input_shape}'
                    ' input channels.'.format(input_shape=input_shape[0]))
            default_shape = (input_shape[0], default_size, default_size)
        else:
            if input_shape[-1] not in {1, 3}:
                warnings.warn(
                    'This model usually expects 1 or 3 input channels. '
                    'However, it was passed an input_shape with {n_input_channels}'
                    ' input channels.'.format(n_input_channels=input_shape[-1]))
            default_shape = (default_size, default_size, input_shape[-1])
    else:
        if data_format == 'channels_first':
            default_shape = (3, default_size, default_size)
        else:
            default_shape = (default_size, default_size, 3)
    if weights == 'imagenet' and require_flatten:
        if input_shape is not None:
            if input_shape != default_shape:
                raise ValueError('When setting `include_top=True` '
                                 'and loading `imagenet` weights, '
                                 '`input_shape` should be {default_shape}.'.format(default_shape=default_shape))
        return default_shape
    if input_shape:
        if data_format == 'channels_first':
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError(
                        '`input_shape` must be a tuple of three integers.')
                if input_shape[0] != 3 and weights == 'imagenet':
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape={input_shape}`'.format(input_shape=input_shape))
                if ((input_shape[1] is not None and input_shape[1] < min_size) or
                    (input_shape[2] is not None and input_shape[2] < min_size)):
                    raise ValueError('Input size must be at least {min_size}x{min_size};'
                                     ' got `input_shape={input_shape}`'.format(min_size=min_size,
                                                                               input_shape=input_shape))
        else:
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError(
                        '`input_shape` must be a tuple of three integers.')
                if input_shape[-1] != 3 and weights == 'imagenet':
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape={input_shape}`'.format(input_shape=input_shape))
                if ((input_shape[0] is not None and input_shape[0] < min_size) or
                    (input_shape[1] is not None and input_shape[1] < min_size)):
                    raise ValueError('Input size must be at least {min_size}x{min_size};'
                                     ' got `input_shape={input_shape}`'.format(min_size=min_size,
                                                                               input_shape=input_shape))
    else:
        if require_flatten:
            input_shape = default_shape
        else:
            if data_format == 'channels_first':
                input_shape = (3, None, None)
            else:
                input_shape = (None, None, 3)
    if require_flatten:
        if None in input_shape:
            raise ValueError('If `include_top` is True, '
                             'you should specify a static `input_shape`. '
                             'Got `input_shape={input_shape}`'.format(input_shape=input_shape))
    return input_shape


def squeeze_excite_block(input_tensor, ratio=16):
    """ Create a channel-wise squeeze-excite block
    Args:
        input_tensor: input Keras tensor
        ratio: number of output filters
    Returns: a Keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    """
    init = input_tensor
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = _tensor_shape(init)[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


def spatial_squeeze_excite_block(input_tensor):
    """ Create a spatial squeeze-excite block
    Args:
        input_tensor: input Keras tensor
    Returns: a Keras tensor
    References
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    """

    se = Conv2D(1, (1, 1), activation='sigmoid', use_bias=False,
                kernel_initializer='he_normal')(input_tensor)

    x = multiply([input_tensor, se])
    return x


def channel_spatial_squeeze_excite(input_tensor, ratio=16):
    """ Create a spatial squeeze-excite block
    Args:
        input_tensor: input Keras tensor
        ratio: number of output filters
    Returns: a Keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    """

    cse = squeeze_excite_block(input_tensor, ratio)
    sse = spatial_squeeze_excite_block(input_tensor)

    x = add([cse, sse])
    return x

def _tensor_shape(tensor):
    return getattr(tensor, '_keras_shape')

def get_model(data_in, data_out, dropout_rate, nb_cnn2d_filt, f_pool_size, t_pool_size,
              rnn_size, fnn_size, weights, doa_objective, baseline, ratio):
    # model definition
    spec_start = Input(shape=(data_in[-3], data_in[-2], data_in[-1]))

    # CNN
    spec_cnn = spec_start
    for i, convCnt in enumerate(f_pool_size):
        
        if baseline is False:

            spec_aux = spec_cnn
            spec_cnn = Conv2D(nb_cnn2d_filt, 3, padding='same')(spec_cnn)
            spec_cnn = BatchNormalization()(spec_cnn)
            spec_cnn = ELU()(spec_cnn) 
            spec_cnn = Conv2D(nb_cnn2d_filt, 3, padding='same')(spec_cnn)
            spec_cnn = BatchNormalization()(spec_cnn) 
        
            spec_aux = Conv2D(nb_cnn2d_filt, 1, padding='same')(spec_aux)
            spec_aux = BatchNormalization()(spec_aux)

            spec_cnn = add([spec_cnn,spec_aux])
            spec_cnn = ELU()(spec_cnn)
            
            if ratio != 0:

                spec_cnn = channel_spatial_squeeze_excite(spec_cnn,ratio=ratio)

                spec_cnn = add([spec_cnn, spec_aux])

        else:

            spec_cnn = Conv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding='same')(spec_cnn)
            spec_cnn = BatchNormalization()(spec_cnn)
            spec_cnn = Activation('relu')(spec_cnn)
        spec_cnn = MaxPooling2D(pool_size=(t_pool_size[i], f_pool_size[i]))(spec_cnn)
        spec_cnn = Dropout(dropout_rate)(spec_cnn)
    spec_cnn = Permute((2, 1, 3))(spec_cnn)

    # RNN
    spec_rnn = Reshape((data_out[0][-2], -1))(spec_cnn)
    for nb_rnn_filt in rnn_size:
        spec_rnn = Bidirectional(
            GRU(nb_rnn_filt, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate,
                return_sequences=True),
            merge_mode='mul'
        )(spec_rnn)

    # FC - DOA
    doa = spec_rnn
    for nb_fnn_filt in fnn_size:
        doa = TimeDistributed(Dense(nb_fnn_filt))(doa)
        doa = Dropout(dropout_rate)(doa)

    doa = TimeDistributed(Dense(data_out[1][-1]))(doa)
    doa = Activation('tanh', name='doa_out')(doa)

    # FC - SED
    sed = spec_rnn
    for nb_fnn_filt in fnn_size:
        sed = TimeDistributed(Dense(nb_fnn_filt))(sed)
        sed = Dropout(dropout_rate)(sed)
    sed = TimeDistributed(Dense(data_out[0][-1]))(sed)
    sed = Activation('sigmoid', name='sed_out')(sed)

    model = None
    if doa_objective is 'mse':
        model = Model(inputs=spec_start, outputs=[sed, doa])
        model.compile(optimizer=Adam(), loss=['binary_crossentropy', 'mse'], loss_weights=weights)
    elif doa_objective is 'masked_mse':
        doa_concat = Concatenate(axis=-1, name='doa_concat')([sed, doa])
        model = Model(inputs=spec_start, outputs=[sed, doa_concat])
        model.compile(optimizer=Adam(), loss=['binary_crossentropy', masked_mse], loss_weights=weights)
    else:
        print('ERROR: Unknown doa_objective: {}'.format(doa_objective))
        exit()
    model.summary()
    return model


def masked_mse(y_gt, model_out):
    # SED mask: Use only the predicted DOAs when gt SED > 0.5
    sed_out = y_gt[:, :, :14] >= 0.5 #TODO fix this hardcoded value of number of classes
    sed_out = keras.backend.repeat_elements(sed_out, 3, -1)
    sed_out = keras.backend.cast(sed_out, 'float32')

    # Use the mask to computed mse now. Normalize with the mask weights #TODO fix this hardcoded value of number of classes
    return keras.backend.sqrt(keras.backend.sum(keras.backend.square(y_gt[:, :, 14:] - model_out[:, :, 14:]) * sed_out))/keras.backend.sum(sed_out)


def load_seld_model(model_file, doa_objective):
    if doa_objective is 'mse':
        return load_model(model_file)
    elif doa_objective is 'masked_mse':
        return load_model(model_file, custom_objects={'masked_mse': masked_mse})
    else:
        print('ERROR: Unknown doa objective: {}'.format(doa_objective))
        exit()



