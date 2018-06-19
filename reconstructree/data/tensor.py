from reconstructree.data.patch import *


class Tensor:

    tensor = array([])
    shape = array([])
    voxelsize = array([])
    bbx = array([])

    def __init__(self, tensor, shape, voxelsize, bbx):
        self.values = array(tensor)
        self.shape = array(shape)
        self.voxelsize = array(voxelsize)
        self.bbx = array(bbx)

    @classmethod
    def from_pointset(cls, pointset, voxelsize, bbx):
        tensorshape = ((array(bbx[1]) - bbx[0]) / voxelsize + 1).astype(int)
        tensor = zeros(tensorshape, dtype=int)
        tensor[tuple(((array(pointset) - bbx[0]) / voxelsize).astype(int).T)] = 1
        return cls(tensor, tensorshape, voxelsize, bbx)

    def to_pointset(self, threshold=0):
        pointset = [array(i) * self.voxelsize + self.bbx[0] for i, v in ndenumerate(self.tensor) if v > threshold]
        return pointset

    def get_patch(self, origin, patchsize):
        return Patch.from_tensor(self.tensor, origin, patchsize)

    def get_patches(self, origins, patchsize):
        return [self.get_patch(origin, patchsize) for origin in origins]

    def get_random_patch(self, patchsize, threshold=None):
        patch = self.get_patch(self.random_origin(), patchsize)
        if threshold:
            while count_nonzero(patch.tensor) < threshold:
                patch = self.get_patch(self.random_origin(), patchsize)
        return patch

    def get_random_patches(self, nbpatches, patchsize, threshold=None):
        return [self.get_random_patch(patchsize, threshold=threshold) for i in range(nbpatches)]

    def get_regular_patches(self, patchsize, stride=None):
        s = patchsize if not stride else ((stride,) * len(patchsize) if isinstance(stride, int) else stride)
        return self.get_patches(self.regular_origins(s), patchsize)

    def random_origin(self):
        return array([random.randint(0, d) for d in self.shape])

    def random_origins(self, nborigins):
        return array([self.random_origin() for i in range(nborigins)])

    def regular_origins(self, stride=None):
        s = (stride,) * len(self.shape) if isinstance(stride, int) else stride
        r = [arange(0, v, s[i]) for i, v in enumerate(self.shape)]
        o = array(meshgrid(*r)).T.reshape(-1, len(self.shape))
        return o[lexsort(transpose(o)[::-1])]


def get_same_patch(tensors, origin, patchsize):
    return [tensor.get_patch(origin, patchsize) for tensor in tensors]


def get_same_patches(tensors, origins, patchsize):
    return [get_same_patch(tensors, origin, patchsize) for origin in origins]
    # array(return).T ?


def get_same_random_patch(tensors, patchsize, threshold=None, targets=None):
    t = [targets] if isinstance(targets, int) else targets
    if threshold and targets:
        patches = get_same_patch(tensors[t], tensors[t[0]].random_origin(), patchsize)
        while any([count_nonzero(patch.tensor) < threshold for patch in patches]):
            patches = get_same_patch(tensors[t], tensors[t[0]].random_origin(), patchsize)
    else: patches = get_same_patch(tensors[0].random_origin())
    return patches


def get_same_random_patches(tensors, nbpatches, patchsize, threshold=None, targets=None):
    t = [targets] if isinstance(targets, int) else targets
    return [get_same_random_patch(tensors, patchsize, threshold=threshold, targets=targets) for i in range(nbpatches)]
    # array(return).T ?


def get_same_regular_patches(tensors, patchsize, stride=None, target=None):
    return get_same_patches(tensors, tensors[0 if target is None else target].regular_origins(stride), patchsize)
