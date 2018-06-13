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


def patches_from_nn_format(output, origins):
    squeeze(output, axis=len(shape(output)))


def totensor(patches, tensorshape):
    tensorlist = full(tensorshape, type("", (), dict(list=[0])))
    for p in patches:
        for i, v in ndenumerate(p.tensor):
            tensorlist[p.origin + i].list.append(v)
    tensor = zeros(tensorshape)
    for i, obj in ndenumerate(tensorlist):
        tensor[i] = sum(obj.list) / len(obj.list)
    return tensor
