from numpy import *
from reconstructree.data.tensor import *


# return the bounding box of the given pointset

def getbbx(pointset):
    return array([amin(array(pointset), axis=0), amax(array(pointset), axis=0)])


# remove the specified "cut" dimension from the pointset

def cutpointset(arr, cut, axis):
    return delete(array(arr), cut, axis)

# return an array containing all the points with the "axis" coord between the 2 values of the slice
# ex : slicepointset(arr, (-1, 1), axis=1) return an array of all points having -1 <= y coord <= 1

def slicepointset(arr, slice, axis):
    return array([point for point in arr if slice[0] <= point[axis] <= slice[1]])


# voxelise a pointset and return the corresponding tensor

def totensor(pointset, voxelsize, bbx=None):
    return Tensor.from_pointset(pointset, voxelsize, bbx)


# voxelise several pointsets and return an array of the corresponding tensors

def to_tensors(pointsets, bbx, voxelsize):
    return [Tensor.from_pointset(pointset, voxelsize, bbx) for pointset in pointsets]
