from numpy import *


class Patch:

    tensor = array([])
    origin = array([])

    def __init__(self, tensor, origin, patchsize):
        patch = array(tensor[[slice(origin[i], origin[i] + patchsize[i]) for i, v in enumerate(shape(tensor))]])
        self.tensor = pad(patch, [(0, patchsize[i] - d) for i, d in enumerate(shape(patch))], 'constant', constant_values=0)
        self.origin = origin

    def to_nn_format(self, dtype=float):
        return self.tensor[..., newaxis].astype(dtype)


def toinput(patches):
    return array([patch.to_nn_format() for patch in patches])