from numpy import *


class Patch:

    tensor = array([])
    origin = array([])

    def __init__(self, tensor, origin):
        self.tensor = array(tensor)
        self.origin = array(origin)

    @classmethod
    def from_tensor(cls, origintensor, origin, patchsize):
        patch = array(origintensor[[slice(origin[i], origin[i] + patchsize[i]) for i, v in enumerate(shape(origintensor))]])
        tensor = pad(patch, [(0, patchsize[i] - d) for i, d in enumerate(shape(patch))], 'constant', constant_values=0)
        return cls(tensor, origin)

    @classmethod
    def from_nn_format(cls, output, origin):
        tensor = squeeze(output, axis=len(shape(output)) - 1)
        return cls(tensor, origin)

    def to_nn_format(self, dtype=float):
        return self.tensor[..., newaxis].astype(dtype)


def toinput(patches):
    return array([p.to_nn_format() for p in patches])


def patches_from_nn_format(outputs, origins):
    return array([Patch.from_nn_format(v, origins[i]) for i, v in enumerate(outputs)])


def tensor_from_patches(patches, tensorshape):
    tensorlist = empty(tensorshape, dtype=object)
    for i, v in ndenumerate(tensorlist): tensorlist[i] = type("numpylist", (), dict(list=[]))
    for p in patches:
        for i, v in ndenumerate(p.tensor):
            o = tuple(p.origin + i)
            oint = all(greater_equal(o, 0)) and all(greater(tensorshape, o))
            #print(o, oint)
            if oint:
                tensorlist[o].list.append(v)
    tensor = zeros(tensorshape)
    for i, v in ndenumerate(tensorlist):
        if v.list: tensor[i] = sum(v.list) / len(v.list)
    return tensor
