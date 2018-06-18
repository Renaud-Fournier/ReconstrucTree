from reconstructree.data.tensors import *
from reconstructree.data.pointsets import *
from reconstructree.data.patches import *


def postprocesspredict(predictions, origins, tensorshape, voxelsize, bbx):
    predpatches = patches_from_nn_format(predictions, origins)
    predtensor = fast_tensor_from_patches(predpatches, tensorshape)
    predpointset = topointset(predtensor, voxelsize, bbx, threshold=.05)
    return predpointset