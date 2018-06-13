from keras.models import Model
from keras.layers import *


def metalayer(dim, layers, depht=1, degree=1, kernel=32):
    for d in range(degree): layers.append(globals()["Conv{}D".format(dim)](kernel, 3, activation='relu', padding='same')(layers[-1]))
    if depht:
        layers.append(globals()["MaxPooling{}D".format(dim)](pool_size=2)(layers[-1]))
        metalayer(dim, layers, depht - 1, degree, kernel * 2)
        layers.append(globals()["UpSampling{}D".format(dim)](size=2)(layers[-1]))
        layers.append(concatenate([layers[-1], layers[-(depht*(3+2*degree)-degree)]], axis=dim+1))
        for d in range(degree): layers.append(globals()["Conv{}D".format(dim)](kernel, 3, activation='relu', padding='same')(layers[-1]))


def metamodel(patchsize, depht=1, degree=1, kernel=32):
    dim = len(patchsize)
    layers = [Input((*patchsize, 1))]
    metalayer(dim, layers, depht=depht, degree=degree, kernel=kernel)
    layers.append(globals()["Conv{}D".format(dim)](1, 1, activation='sigmoid')(layers[-1]))
    model = Model(layers[0], layers[-1])
    return model


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