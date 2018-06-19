from keras.models import Model
from keras.layers import *


# return a encoder decoder model
# depth is the number of image reduction layers (and of corresponding image augmentation layers)
# degree is the number of successive convolution layers in every convolution phase
# kernel is ths number of kernel in the convolution phases of layer 0 (divided by 2 at each depth level

def enco_deco_model(patchsize, depth=1, degree=1, kernel=32):
    dim = len(patchsize)
    layers = [Input((*patchsize, 1))]
    enco_deco_layer(dim, layers, depth=depth, degree=degree, kernel=kernel)
    layers.append(globals()["Conv{}D".format(dim)](1, 1, activation='sigmoid')(layers[-1]))
    model = Model(layers[0], layers[-1])
    return model


# recursively create the depht layers used in the enco_deco_model

def enco_deco_layer(dim, layers, depth=1, degree=1, kernel=32):
    for d in range(degree): layers.append(globals()["Conv{}D".format(dim)](kernel, 3, activation='relu', padding='same')(layers[-1]))
    if depth:
        layers.append(globals()["MaxPooling{}D".format(dim)](pool_size=2)(layers[-1]))
        enco_deco_layer(dim, layers, depth - 1, degree, kernel * 2)
        layers.append(globals()["UpSampling{}D".format(dim)](size=2)(layers[-1]))
        layers.append(concatenate([layers[-1], layers[-(depth*(3+2*degree)-degree)]], axis=dim+1))
        for d in range(degree): layers.append(globals()["Conv{}D".format(dim)](kernel, 3, activation='relu', padding='same')(layers[-1]))


# return a simple encoder-decoder model

def examplemodel(patchsize):

    inputs = Input((*patchsize, 1))

    conv1 = Conv3D(32, 3, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling3D(pool_size=2)(conv1)

    conv2 = Conv3D(64, 3, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling3D(pool_size=2)(conv2)

    conv3 = Conv3D(128, 3, activation='relu', padding='same')(pool2)

    up4 = UpSampling3D(size=2)(conv3)
    concat4 = concatenate([up4, conv2], axis=4)
    conv4 = Conv3D(64, 3, activation='relu', padding='same')(concat4)

    up5 = UpSampling3D(size=2)(conv4)
    concat5 = concatenate([up5, conv1], axis=4)
    conv5 = Conv3D(32, 3, activation='relu', padding='same')(concat5)

    outputs = Conv3D(1, 1, activation='sigmoid')(conv5)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model


def basicmodel(patchsize):

    couche_entree = Input((*patchsize, 1))

    couche_sortie = Conv3D(1, 1, activation='sigmoid')(couche_entree)

    model = Model(inputs=[couche_entree], outputs=[couche_sortie])

    return model