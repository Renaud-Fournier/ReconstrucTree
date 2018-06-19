from reconstructree.data.tensors import *
from reconstructree.data.patch import *


# return a pointset created from neural network output
# must be given the predictions, the corresponding origins, a tensor shape, a voxelsize and a bounding box

def postprocesspredict(predictions, origins, tensorshape, voxelsize, bbx):
    predpatches = patches_from_nn_format(predictions, origins)
    predtensor = fast_tensor_from_patches(predpatches, tensorshape)
    predpointset = topointset(predtensor, voxelsize, bbx, threshold=.05)
    return predpointset