from keras.layers import BatchNormalization, Dropout, Flatten, Dense, Activation
import numpy as np


# Function for creating a dense layer(s)
def dense_block(layer, n_neurons, flatten=True, inner_activation='relu', last_activation='sigmoid', batch_norm=True,
                batch_norm_after_activation=True, dropout=None, name='dense'):
    # Check input
    assert isinstance(n_neurons, (tuple, list)) and len(n_neurons) > 0, 'Input dense layers must be tuple or list of ' \
                                                                        'at least 1 elements, typically (n, 1) or (1)'
    # Check input dropout
    if dropout is None:
        dropout = list(np.zeros(len(n_neurons)-1))
    assert isinstance(dropout, (tuple, list)) and len(dropout) == len(n_neurons)-1, \
        'Input dense layers must be tuple or list of the same length as n_neurons-1'

    if flatten:
        layer = Flatten()(layer)
    # Add inner layers
    if batch_norm:
        for i, (n, d) in enumerate(zip(n_neurons[:-1], dropout)):
            layer = Dense(n, name='{}{}'.format(name, i+1))(layer)
            if batch_norm_after_activation:
                layer = Activation(inner_activation)(layer)
                if d > 0:
                    layer = Dropout(d, name='{}_dropout{}_{}'.format(name, i+1, d))(layer)
                layer = BatchNormalization(name='{}_bn{}'.format(name, i+1))(layer)
            else:
                layer = BatchNormalization(name='{}_bn{}'.format(name, i+1))(layer)
                layer = Activation(inner_activation)(layer)
                if d > 0:
                    layer = Dropout(d, name='{}_dropout{}_{}'.format(name, i+1, d))(layer)
    else:
        for i, (n, d) in enumerate(zip(n_neurons[:-1], dropout)):
            layer = Dense(n, activation=inner_activation, name='{}{}'.format(name, i + 1))(layer)
            if d > 0:
                layer = Dropout(d, name='{}_dropout{}_{}'.format(name, i+1, d))(layer)

    # Add the last dense layer - the prediction layer
    layer = Dense(n_neurons[-1], activation=last_activation, name='predictions')(layer)
    return layer
