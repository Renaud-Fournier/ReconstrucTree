from reconstructree.data.patches import *


# run through the entire tensor and regularly extract patches of dimension patchsize
# patches are extracted every (x,y,z) steps as precised in stride
# if not precised, stride equals patchsize (patches are adjacent)

def regularpatches(tensor, patchsize, stride=None):
    s = patchsize if not stride else (
        (stride,) * len(patchsize) if isinstance(stride, int) else stride)
    o = regularorigins(shape(tensor), s)
    return patches(tensor, o, patchsize)


# return a patch extracted at a random position containing at least threshold non-zero voxels

def randpatch(tensor, patchsize, threshold=None):
    o = randorigin(shape(tensor))
    p = Patch.from_tensor(tensor, o, patchsize)
    if threshold:
        while count_nonzero(p.tensor) < threshold:
            o = randorigin(shape(tensor))
            p = Patch.from_tensor(tensor, o, patchsize)
    return p


# return an array of patches extracted at the same given position in several tensors

def samepatch(tensors, origin, patchsize):
    return [Patch.from_tensor(tensor, origin, patchsize) for tensor in tensors]


# return an array of patches extracted at the same random position in several tensors
# the patch extracted in the target tensor must have at least threshold non-zero voxels

def samerandpatch(tensors, patchsize, threshold=None, target=0):
    o = randpatch(tensors[target], patchsize, threshold=threshold).origin
    return samepatch(tensors, o, patchsize)


# return the patches extracted at the given positions in the tensor

def patches(tensor, origins, patchsize):
    return [Patch.from_tensor(tensor, origin, patchsize) for origin in origins]


# return nbpatches randomly extracted patches from a given tensor according to a minimal number of non-zero voxels

def randpatches(tensor, nbpatches, patchsize, threshold=None):
    return [randpatch(tensor, patchsize, threshold=threshold) for i in range(nbpatches)]


# return array of array of patches extracted at the same given origins in each tensor of tensors

def samepatches(tensors, origins, patchsize):
    return [patches(tensor, origins, patchsize) for tensor in tensors]


# return array of array of patches extracted at the same random origins in each tensor of tensors

def samerandpatches(tensors, nbpatches, patchsize, threshold=None, target=0):
    return [samerandpatch(tensors, patchsize, threshold=threshold, target=target) for i in nbpatches]


# return a random origin in the given tensor shape

def randorigin(tensorshape):
    return array([random.randint(0, d) for d in tensorshape])


# return  nborigins origins in the given tensor shape

def randorigins(tensorshape, nborigins):
    return array([randorigin(tensorshape) for i in range(nborigins)])


# return origins regularly extracted according to the given stride in the tensor shape

def regularorigins(tensorshape, stride, sorted=True):
    s = (stride,) * len(tensorshape) if isinstance(stride, int) else stride
    r = [arange(0, v, s[i]) for i, v in enumerate(tensorshape)]
    o = array(meshgrid(*r)).T.reshape(-1, len(tensorshape))
    return o[lexsort(transpose(o)[::-1])] if sorted else o


# convert a tensor to a pointset

def topointset(tensor, voxelsize, bbx, threshold=0):
    pointset = [array(i) * voxelsize + bbx[0] for i, v in ndenumerate(tensor) if v > threshold]
    return pointset
