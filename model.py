#!/usr/bin/env python3

from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, Activation, MaxPooling2D, Dropout, BatchNormalization
from keras.layers import add, concatenate
from keras.losses import binary_crossentropy
from keras.optimizers import Adam

from metrics import f1

def conv2d_block(inputs, n_filter, kernel_size=3, batchnorm=True, activation='relu'):
    # first layer
    x = Conv2D(n_filter, kernel_size=kernel_size, kernel_initializer="he_normal",
               padding="same")(inputs)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)
    # second layer
    x = Conv2D(n_filter, kernel_size=kernel_size, kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x


def unet(pretrained_weights = None,
         input_size = (None,None,3),
         n_filter=16,
         activation='relu',
         dropout=True, dropout_rate=0.5,
         batchnorm=True,
         loss=binary_crossentropy,
         optimizer=Adam(lr=1e-4)):
    """Build a standard UNet model.
    
    Arguments:
        pretrained_weights {str} -- path of the pretrained weights (default: {None})
        input_size {tuple} -- size of input images (default: {(None,None,3)})
        n_filter {int} -- number of filter of the first layer (default: {16})
        activation {str} -- activation function to use (default: {'relu'})
        dropout {bool} -- whether to use dropout layer (default: {True})
        dropout_rate {float} -- dropout rate (default: {0.5})
        batchnorm {bool} -- whether to use batch normalization layer (default: {True})
        loss {keras.losses} -- loss function to use (default: {binary_crossentropy})
        optimizer {keras.optimizers} -- optimizer to use (default: {Adam(lr=1e-4)})
    
    Returns:
        keras.models -- UNet model
    """

  
    # 3
    inputs = Input(input_size)
    
    # down path
    # n_filter
    conv1 = conv2d_block(inputs, n_filter, kernel_size=3, batchnorm=batchnorm, activation=activation)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # n_filter*2
    conv2 = conv2d_block(pool1, n_filter*2, kernel_size=3, batchnorm=batchnorm, activation=activation)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # n_filter*4
    conv3 = conv2d_block(pool2, n_filter*4, kernel_size=3, batchnorm=batchnorm, activation=activation)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # n_filter*8
    conv4 = conv2d_block(pool3, n_filter*8, kernel_size=3, batchnorm=batchnorm, activation=activation)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # central path
    # n_filter*16
    conv5 = conv2d_block(pool4, n_filter*16, kernel_size=3, batchnorm=batchnorm, activation=activation)

    # up path
    # n_filter*8
    up6 = Conv2DTranspose(n_filter*8, kernel_size=2, strides=2, kernel_initializer="he_normal", padding='same')(conv5)
    merge6 = concatenate([conv4, up6], axis = 3)
    merge6 = Dropout(dropout_rate)(merge6) if dropout else merge6
    conv6 = conv2d_block(merge6, n_filter*8, kernel_size=3, batchnorm=False, activation=activation)
    
    # n_filter*4
    up7 = Conv2DTranspose(n_filter*4, kernel_size=2, strides=2, kernel_initializer="he_normal", padding='same')(conv6)
    merge7 = concatenate([conv3, up7], axis = 3)
    merge7 = Dropout(dropout_rate)(merge7) if dropout else merge7
    conv7 = conv2d_block(merge7, n_filter*4, kernel_size=3, batchnorm=False, activation=activation)
    
    # n_filter*2
    up8 = Conv2DTranspose(n_filter*2, kernel_size=2, strides=2, kernel_initializer="he_normal", padding='same')(conv7)
    merge8 = concatenate([conv2, up8], axis = 3)
    merge8 = Dropout(dropout_rate)(merge8) if dropout else merge8
    conv8 = conv2d_block(merge8, n_filter*2, kernel_size=3, batchnorm=False, activation=activation)
    
    # n_filter
    up9 = Conv2DTranspose(n_filter, kernel_size=2, strides=2, kernel_initializer="he_normal", padding='same')(conv8)
    merge9 = concatenate([conv1, up9], axis = 3)
    merge9 = Dropout(dropout_rate)(merge9) if dropout else merge9
    conv9 = conv2d_block(merge9, n_filter, kernel_size=3, batchnorm=False, activation=activation)
    
    # classifier
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = optimizer, loss = loss, metrics = [f1, 'accuracy'])
    
    if(pretrained_weights):
        model.load_weights(filepath=pretrained_weights)

    return model



def bottleneck(x, n_filter, depth=6, kernel_size=3, activation='relu'):
    """Bottle neck of UNet with dilated convolution."""
    dilated_layers = []
    for i in range(depth):
        x = Conv2D(n_filter, kernel_size, 
                    activation=activation, padding='same', dilation_rate=2**i)(x)
        dilated_layers.append(x)
    return add(dilated_layers)

def unet_dilated(pretrained_weights = None,
                 input_size = (None,None,3),
                 n_filter=16,
                 activation='relu',
                 dropout=True, dropout_rate=0.5,
                 batchnorm=True,
                 loss=binary_crossentropy,
                 optimizer=Adam(lr=1e-4)):
    """Build a standard UNet model with dilated convolution.
    
    Arguments:
        pretrained_weights {str} -- path of the pretrained weights (default: {None})
        input_size {tuple} -- size of input images (default: {(None,None,3)})
        n_filter {int} -- number of filter of the first layer (default: {16})
        activation {str} -- activation function to use (default: {'relu'})
        dropout {bool} -- whether to use dropout layer (default: {True})
        dropout_rate {float} -- dropout rate (default: {0.5})
        batchnorm {bool} -- whether to use batch normalization layer (default: {True})
        loss {keras.losses} -- loss function to use (default: {binary_crossentropy})
        optimizer {keras.optimizers} -- optimizer to use (default: {Adam(lr=1e-4)})
    
    Returns:
        keras.models -- UNet model with dilated conv bottlenect
    """
    # 3
    inputs = Input(input_size)
    
    # down path
    # n_filter
    conv1 = conv2d_block(inputs, n_filter, kernel_size=3, batchnorm=batchnorm, activation=activation)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # n_filter*2
    conv2 = conv2d_block(pool1, n_filter*2, kernel_size=3, batchnorm=batchnorm, activation=activation)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # n_filter*4
    conv3 = conv2d_block(pool2, n_filter*4, kernel_size=3, batchnorm=batchnorm, activation=activation)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # central path
    # n_filter*8
    dilated = bottleneck(pool3, n_filter*8, depth=6, kernel_size=3, activation=activation)

    # up path
    # n_filter*4
    up4 = Conv2DTranspose(n_filter*4, kernel_size=2, strides=2, kernel_initializer="he_normal", padding='same')(dilated)
    merge4 = concatenate([conv3, up4], axis = 3)
    merge4 = Dropout(dropout_rate)(merge4) if dropout else merge4
    conv4 = conv2d_block(merge4, n_filter*4, kernel_size=3, batchnorm=False, activation=activation)
    
    # n_filter*2
    up5 = Conv2DTranspose(n_filter*2, kernel_size=2, strides=2, kernel_initializer="he_normal", padding='same')(conv4)
    merge5 = concatenate([conv2, up5], axis = 3)
    merge5 = Dropout(dropout_rate)(merge5) if dropout else merge5
    conv5 = conv2d_block(merge5, n_filter*2, kernel_size=3, batchnorm=False, activation=activation)
    
    # n_filter
    up6 = Conv2DTranspose(n_filter, kernel_size=2, strides=2, kernel_initializer="he_normal", padding='same')(conv5)
    merge6 = concatenate([conv1, up6], axis = 3)
    merge6 = Dropout(dropout_rate)(merge6) if dropout else merge6
    conv6 = conv2d_block(merge6, n_filter, kernel_size=3, batchnorm=False, activation=activation)
    
    # classifier
    conv7 = Conv2D(1, 1, activation='sigmoid')(conv6)

    model = Model(inputs = inputs, outputs = conv7)

    model.compile(optimizer = optimizer, loss = loss, metrics = [f1, 'accuracy'])
    
    if(pretrained_weights):
        model.load_weights(filepath=pretrained_weights)

    return model
