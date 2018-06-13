from reconstructree.data.patches import *


def randpatch(tensor, patchsize, threshold=None):
    o = randorigin(shape(tensor))
    p = Patch(tensor, o, patchsize)
    if threshold:
        while count_nonzero(p.tensor) < threshold:
            o = randorigin(shape(tensor))
            p = Patch(tensor, o, patchsize)
    return p


def samepatch(tensors, origin, patchsize):
    return [Patch(tensor, origin, patchsize) for tensor in tensors]


def samerandpatch(tensors, patchsize, threshold=None, target=0):
    o = randpatch(tensors[target], patchsize, threshold=threshold).origin
    return samepatch(tensors, o, patchsize)


def patches(tensor, origins, patchsize):
    return [Patch(tensor, origin, patchsize) for origin in origins]


def randpatches(tensor, nbpatches, patchsize, threshold=None):
    return [randpatch(tensor, patchsize, threshold=threshold) for i in range(nbpatches)]


def samepatches(tensors, origins, patchsize):
    return [patches(tensor, origins, patchsize) for tensor in tensors]


def samerandpatches(tensors, nbpatches, patchsize, threshold=None, target=0):
    return [samerandpatch(tensors, patchsize, threshold=threshold, target=target) for i in nbpatches]


def randorigin(tensorshape):
    return array([random.randint(0, d) for d in tensorshape])


def randorigins(tensorshape, nborigins):
    return array([randorigin(tensorshape) for i in range(nborigins)])