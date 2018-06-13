from numpy import *


def getbbx(set):
    return array([amin(array(set), axis=0), amax(array(set), axis=0)])


def cutset(arr, cut, axis=1):
    return delete(array(arr), cut, axis)


def sliceset(arr, slice, axis=0):
    return array([v for v in arr if slice[0] <= v[axis] <= slice[1]])


def totensor(pointset, bbx, voxelsize, dtype=int):
    tensorshape = ((array(bbx[1]) - bbx[0]) / voxelsize + 1).astype(int)
    tensor = zeros(tensorshape, dtype=dtype)
    tensor[tuple(((array(pointset) - bbx[0]) / voxelsize).astype(int).T)] = 1
    return tensor


def totensors(sets, bbx, voxelsize, dtype=int):
    return [totensor(set, bbx, voxelsize, dtype=dtype) for set in sets]