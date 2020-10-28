"""Inception V3 model for 1D inputs. Coded according to the [kentsommer GitHub implemenation](
    https://github.com/kentsommer/keras-inceptionV4/blob/master/inception_v4.py).
    Inception V4 model for 1D inputs. Coded according to the [Keras implementation](
    https://github.com/keras-team/keras-applications/blob/master/keras_applications/inception_v3.py).
Reference
- [Rethinking the Inception Architecture for Computer Vision](
    http://arxiv.org/abs/1512.00567) (CVPR 2016)
- [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)
"""
import os.path
from keras.models import Model
from keras.layers import Conv1D, BatchNormalization, Activation, Input, MaxPooling1D, AveragePooling1D, concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D, Dense, Dropout
from keras import regularizers
from keras import initializers
# noinspection PyPep8Naming
from keras import backend as K
from keras.utils import get_source_inputs
from densenet import dense_block


# Utility function to apply convolution + batch normalization
def conv1d_bn(x, filters, kernel_size, padding='same', strides=1, use_bias=False, kernel_initializer=None,
              kernel_regularizer=None, bn_momentum=0.99, name=None):
    """Utility function to apply conv + BN.
    Args:
        x: input tensor.
        filters: filters in `Conv1D`.
        kernel_size: size of the convolution kernel.
        padding: padding mode in `Conv1D`.
        strides: strides in `Conv1D`.
        bn_momentum: momentum in `BatchNormalization`.
        use_bias: whether to use bias in `Conv1D`.
        kernel_regularizer:
        kernel_initializer:
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    Returns:
        Output tensor after applying `Conv1D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if kernel_initializer is None:
        kernel_initializer = 'glorot_uniform'
    x = Conv1D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias,
               kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, name=conv_name)(x)
    x = BatchNormalization(scale=False, momentum=bn_momentum, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


###############################################################################
#   Inception V3 modules                                                      #
###############################################################################

# Function for creating a naive inception module V1 of GoogLeNet
def naive_inception_module(layer_in, f1, f2, f3, double=False, pooling='max', name=None):
    """
    Args:
        layer_in ():
        f1 ():
        f2 ():
        f3 ():
        double ():
        pooling ():
        name ():

    Returns:
    """
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    # 1x conv
    conv1 = conv1d_bn(layer_in, f1, 1)
    # 3x conv
    conv3 = conv1d_bn(layer_in, f2, 3)
    if double:
        conv3 = conv1d_bn(conv3, f2, 3)
    # 5x conv
    conv5 = conv1d_bn(layer_in, f3, 5)
    # 3x max pooling
    if pooling == 'max':
        pool = MaxPooling1D(3, strides=1, padding='same')(layer_in)
    else:
        pool = AveragePooling1D(3, strides=1, padding='same')(layer_in)
    # Concatenate filters, assumes filters/channels last
    layer_out = concatenate([conv1, conv3, conv5, pool], axis=channel_axis, name=name)
    return layer_out


# function for creating a projected inception module
def inception_module(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out, double=False, pooling='avg', name=None):
    """
    Args:
        layer_in ():
        f1 ():
        f2_in ():
        f2_out ():
        f3_in ():
        f3_out ():
        f4_out ():
        double (bool): Double the middle convolution layer
        pooling (str):
        name ():

    Returns:
        layer_out: Output tensor

    """
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    # 1x conv
    conv1 = conv1d_bn(layer_in, f1, 1)
    # 3x conv
    conv3 = conv1d_bn(layer_in, f2_in, 1)
    conv3 = conv1d_bn(conv3, f2_out, 3)
    if double:
        conv3 = conv1d_bn(conv3, f2_out, 3)
    # 5x conv
    conv5 = conv1d_bn(layer_in, f3_in, 1)
    conv5 = conv1d_bn(conv5, f3_out, 5)
    # 3x max pooling
    if pooling == 'avg':
        pool = AveragePooling1D(3, strides=1, padding='same')(layer_in)
    else:
        pool = MaxPooling1D(3, strides=1, padding='same')(layer_in)
    pool = conv1d_bn(pool, f4_out, 1)
    # concatenate filters, assumes filters/channels last
    layer_out = concatenate([conv1, conv3, conv5, pool], axis=channel_axis, name=name)
    return layer_out


# noinspection PyPep8Naming
def InceptionV31D(input_shape=None, weights=None, input_tensor=None, pooling=None, classes=2, dropout=0.0, top={'n_neurons': [], 'dropout': [], 'batch_norm': True}):
    """Instantiates the Inception v3 architecture.
    Optionally loads pre-trained weights.
    Note that the data format convention used by the model is
    the one specified in your Keras cfg_clb at `~/.keras/keras.json`.
    # Arguments
        input_shape: optional shape tuple, only to be specified
            if `top` is None (otherwise the input shape
            has to be `(...)` (with `channels_last` data format)
            or `(...)` (with `channels_first` data format).
            It should have exactly 3 input channel,
            and width and height should be no smaller than 75.
            E.g. `(150, 1)` would be one valid value.
        weights: one of `None` (random initialization)
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        include_top: whether to include the fully-connected
            layer at the top of the network.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights is not None and not os.path.exists(weights):
        raise ValueError('The `weights` argument should be either'
                         '`None` (random initialization),'
                         'or the path to the weights file to be loaded.')
    
    if input_tensor is None:
        inp = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            inp = Input(tensor=input_tensor, shape=input_shape)
        else:
            inp = input_tensor

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    # Input layer
    inp = Input(shape=input_shape)
    x = conv1d_bn(inp, 32, 3, strides=2, padding='valid')
    x = conv1d_bn(x, 32, 3, padding='valid')
    x = conv1d_bn(x, 64, 3)
    x = MaxPooling1D(3, strides=2)(x)

    x = conv1d_bn(x, 80, 1, padding='valid')
    x = conv1d_bn(x, 192, 3, padding='valid')
    x = MaxPooling1D(3, strides=2)(x)

    # mixed 0: 35 x 35 x 256
    x = inception_module(x, 64, 64, 96, 48, 64, 32, double=True, name='mixed0')
    # mixed 1: 35 x 35 x 288
    x = inception_module(x, 64, 64, 96, 48, 64, 64, double=True, name='mixed1')
    # mixed 2: 35 x 35 x 288
    x = inception_module(x, 64, 64, 96, 48, 64, 64, double=True, name='mixed2')

    # mixed 3: 17 x 17 x 768
    conv3 = conv1d_bn(x, 384, 3, strides=2, padding='valid')

    conv3dbl = conv1d_bn(x, 64, 1)
    conv3dbl = conv1d_bn(conv3dbl, 96, 3)
    conv3dbl = conv1d_bn(conv3dbl, 96, 3, strides=2, padding='valid')

    pool = MaxPooling1D(3, strides=2)(x)
    x = concatenate([conv3, conv3dbl, pool], axis=channel_axis, name='mixed3')

    # mixed 4: 17 x 17 x 768
    conv1 = conv1d_bn(x, 192, 1)

    conv7 = conv1d_bn(x, 128, 1)
    conv7 = conv1d_bn(conv7, 128, 7)

    conv7dbl = conv1d_bn(x, 128, 1)
    conv7dbl = conv1d_bn(conv7dbl, 128, 7)
    conv7dbl = conv1d_bn(conv7dbl, 128, 7)

    pool = AveragePooling1D(3, strides=1, padding='same')(x)
    pool = conv1d_bn(pool, 192, 1)
    x = concatenate([conv1, conv7, conv7dbl, pool], axis=channel_axis, name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        conv1 = conv1d_bn(x, 192, 1)

        conv7 = conv1d_bn(x, 160, 1)
        conv7 = conv1d_bn(conv7, 192, 7)

        conv7dbl = conv1d_bn(x, 160, 1)
        conv7dbl = conv1d_bn(conv7dbl, 160, 7)
        conv7dbl = conv1d_bn(conv7dbl, 192, 7)

        pool = AveragePooling1D(3, strides=1, padding='same')(x)
        pool = conv1d_bn(pool, 192, 1)
        x = concatenate([conv1, conv7, conv7dbl, pool], axis=channel_axis, name='mixed' + str(5+i))

    # mixed 7: 17 x 17 x 768
    conv1 = conv1d_bn(x, 192, 1)

    conv7 = conv1d_bn(x, 192, 1)
    conv7 = conv1d_bn(conv7, 192, 7)

    conv7dbl = conv1d_bn(x, 192, 1)
    conv7dbl = conv1d_bn(conv7dbl, 192, 7)
    conv7dbl = conv1d_bn(conv7dbl, 192, 7)

    pool = AveragePooling1D(3, strides=1, padding='same')(x)
    pool = conv1d_bn(pool, 192, 1)
    x = concatenate([conv1, conv7, conv7dbl, pool], axis=channel_axis, name='mixed7')

    # mixed 8: 8 x 8 x 1280
    conv3 = conv1d_bn(x, 192, 1)
    conv3 = conv1d_bn(conv3, 320, 3, strides=2, padding='valid')

    conv7 = conv1d_bn(x, 192, 1)
    conv7 = conv1d_bn(conv7, 192, 7)
    conv7 = conv1d_bn(conv7, 192, 3, strides=2, padding='valid')

    pool = MaxPooling1D(3, strides=2)(x)
    x = concatenate([conv3, conv7, pool], axis=channel_axis, name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        conv1 = conv1d_bn(x, 320, 1)

        conv3 = conv1d_bn(x, 384, 1)
        conv3 = conv1d_bn(conv3, 384, 3)

        conv3dbl = conv1d_bn(x, 448, 1)
        conv3dbl = conv1d_bn(conv3dbl, 384, 3)
        conv3dbl = conv1d_bn(conv3dbl, 384, 3)

        pool = AveragePooling1D(3, strides=1, padding='same')(x)
        pool = conv1d_bn(pool, 192, 1)
        x = concatenate([conv1, conv3, conv3dbl, pool], axis=channel_axis, name='mixed' + str(9 + i))
    
    if top is None or top['n_neurons'] is None:
        if pooling == 'avg':
            x = GlobalAveragePooling1D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling1D()(x)
    else:
        # Include the fully-connected layer(s) at the top of the network
        # Classification/prediction block
        if classes == 2:
            n_pred_neurons = 1
            last_activation = 'sigmoid'
        else:
            n_pred_neurons = classes
            last_activation = 'softmax'
        dense_layers = list(top['n_neurons'])
        dense_layers.append(n_pred_neurons)
        x = GlobalAveragePooling1D(name='avg_pool')(x)
        if dropout > 0:
            x = Dropout(dropout, name='dropout_{}'.format(dropout))(x)
        x = dense_block(x, dense_layers, flatten=False, last_activation=last_activation, batch_norm=top['batch_norm'], batch_norm_after_activation=True, dropout=top['dropout'], name='fc')
    
    # Ensure that the model takes into account any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = inp
    
    # Create model
    model = Model(inputs, x, name='inceptionV31D')
    
    # Load weights
    if weights is not None:
        model.load_weights(weights)
    
    # Return Keras model instance
    return model
