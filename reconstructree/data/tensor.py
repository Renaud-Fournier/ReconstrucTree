from reconstructree.data.patch import *


# class Tensor : associate an array of values (and its shape) to a voxelsize and a boundingbox

class Tensor:

    def __init__(self, tensor, shape, voxelsize, bbx):
        self.values = array(tensor)  # the array containging the values
        self.shape = array(shape)
        self.voxelsize = array(voxelsize)
        self.bbx = array(bbx)

    @classmethod
    def from_pointset(cls, pointset, voxelsize, bbx=None):

        # create a Tensor from a pointset. if not specifeid, the bbx is the pointset bbx

        true_bbx = array([amin(array(pointset), axis=0), amax(array(pointset), axis=0)]) if bbx is None else bbx
        shape = ceil((array(true_bbx[1]) - true_bbx[0]) / voxelsize).astype(int)
        tensor = zeros(shape, dtype=uint8)
        indices = ((array(pointset) - true_bbx[0]) / voxelsize).astype(int)
        condition = [all(greater_equal(index, 0)) and all(greater(shape, index)) for index in indices]
        for i, v in enumerate(indices[condition]):
            tensor[tuple(v)] = 1
        return cls(tensor, shape, voxelsize, bbx)

    @classmethod
    def from_patches(cls, patches, shape, voxelsize, bbx, func=lambda l: sum(l) / len(l)):

        # create a Tensor from multiple patches by inserting their values in a new tensor of dimensions "shape"
        # due to patches overlapping, voxels can have multiple predictions
        # these predictions are stored for each voxel in a list l to which is applied the function "func"
        # in order to determine the final voxel value

        listtensor = empty(shape, dtype=object)
        for i, v in ndenumerate(listtensor):
            listtensor[i] = type("numpylist", (), dict(list=[]))
        for p in patches:
            for i, v in ndenumerate(p.tensor):
                o = tuple(p.origin + i)
                oint = all(greater_equal(o, 0)) and all(greater(shape, o))
                if oint:
                    listtensor[o].list.append(v)
        tensor = empty(shape)
        for i, v in ndenumerate(listtensor):
            tensor[i] = func(v.list) if v.list else 0
        return cls(tensor, shape, voxelsize, bbx)

    @classmethod
    def from_patches_fast(cls, patches, shape, voxelsize, bbx):

        # create a Tensor from multiple patches by inserting their values in a new tensor of dimensions "shape"
        # already filled voxels are overwritten
        # meant to be used on non-overlaping patches

        tensor = zeros(shape)
        instance = cls(tensor, shape, voxelsize, bbx)
        instance.merge_patches(patches)
        return instance

    def to_pointset(self, threshold=0):

        # return the indices of the voxels having a value >= to the threshold

        pointset = [array(i) * self.voxelsize + self.bbx[0] + (self.voxelsize / 2) for i, v in ndenumerate(self.values) if v > threshold]
        return pointset

    def merge_patch(self, patch):

        # insert a patch in the Tensor (overwrite previous values)

        dim = len(self.shape)
        to, ts, po, ps = (0,) * dim, self.shape, patch.origin, shape(patch.values)
        to, ts, po, ps = map(array, (to, ts, po, ps))
        tbot, ttop, pbot, ptop = maximum(to, po), minimum(ts, po + ps), maximum(to, - po), minimum(ps, - po + ts)
        tslice, pslice = [slice(tbot[i], ttop[i]) for i in range(dim)], [slice(pbot[i], ptop[i]) for i in range(dim)]
        self.values[tslice] = patch.values[pslice]

    def merge_patches(self, patches):

        # insert multiples patches in the Tensor

        for patch in patches:
            self.merge_patch(patch)

    def get_patch(self, origin, patchsize):

        # return a Patch of the Tensor from the specified origin

        return Patch.from_values(self.values, origin, patchsize)

    def get_patches(self, origins, patchsize):

        # return multiple Patches of the Tensor from the specified origins

        return [self.get_patch(origin, patchsize) for origin in origins]

    def get_random_patch(self, patchsize, threshold=None):

        # return a random Patch of the Tensor

        patch = self.get_patch(self.random_origin(), patchsize)
        if threshold:
            while count_nonzero(patch.tensor) < threshold:
                patch = self.get_patch(self.random_origin(), patchsize)
        return patch

    def get_random_patches(self, nbpatches, patchsize, threshold=None):

        # return random Patches of the Tensor

        return [self.get_random_patch(patchsize, threshold=threshold) for i in range(nbpatches)]

    def get_regular_patches(self, patchsize, stride=None):

        # return Patches of the Tensor extracted regularly according to the specified stride
        # if not specified, stride = patchsize

        s = patchsize if not stride else ((stride,) * len(patchsize) if isinstance(stride, int) else stride)
        return self.get_patches(self.regular_origins(s), patchsize)

    def random_origin(self):

        # return a random origin in the Tensor

        return array([random.randint(0, d) for d in self.shape])

    def random_origins(self, nborigins):

        # return random origins in the Tensor

        return array([self.random_origin() for i in range(nborigins)])

    def regular_origins(self, stride):

        # return origins extracted regularly according to the specified stride

        s = (stride,) * len(self.shape) if isinstance(stride, int) else stride
        r = [arange(0, v, s[i]) for i, v in enumerate(self.shape)]
        o = array(meshgrid(*r)).T.reshape(-1, len(self.shape))
        return o[lexsort(transpose(o)[::-1])]


def get_same_patch(tensors, origin, patchsize):

    # return an array of single patch extracted at the same position in each Tensor of "tensors"

    return [tensor.get_patch(origin, patchsize) for tensor in tensors]


def get_same_patches(tensors, origins, patchsize):

    # return an array of patches extracted at the same positions in each Tensor of "tensors"

    return [get_same_patch(tensors, origin, patchsize) for origin in origins]
    # array(return).T ?


def get_same_random_patch(tensors, patchsize, threshold=None, targets=None):

    # return an array of single patch extracted at the same random position in each Tensor of "tensors"

    t = [targets] if isinstance(targets, int) else targets
    if threshold and targets:
        patches = get_same_patch(tensors[t], tensors[t[0]].random_origin(), patchsize)
        while any([count_nonzero(patch.tensor) < threshold for patch in patches]):
            patches = get_same_patch(tensors[t], tensors[t[0]].random_origin(), patchsize)
    else:
        patches = get_same_patch(tensors, tensors[0].random_origin(), patchsize)
    return patches


def get_same_random_patches(tensors, nbpatches, patchsize, threshold=None, targets=None):

    # return an array of patches extracted at the same random positions in each Tensor of "tensors"

    t = [targets] if isinstance(targets, int) else targets
    return [get_same_random_patch(tensors, patchsize, threshold=threshold, targets=targets) for i in range(nbpatches)]
    # array(return).T ?


def get_same_regular_patches(tensors, patchsize, stride=None, target=None):

    # return an array of Patches of the Tensor extracted regularly according to the specified stride in each Tensor of "tensors"
    # if not specified, stride = patchsize

    return get_same_patches(tensors, tensors[0 if target is None else target].regular_origins(stride), patchsize)
