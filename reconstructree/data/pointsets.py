from numpy import *


# return the bounding box of the given pointset

def getbbx(pointset):
    return array([amin(array(pointset), axis=0), amax(array(pointset), axis=0)])


# remove the "cut" dimension in the pointset

def cutpointset(arr, cut, axis=1):
    return delete(array(arr), cut, axis)


# return an array containing the points with the "axis" coord between the 2 values of the slice

def slicepointset(arr, slice, axis=0):
    return array([point for point in arr if slice[0] <= point[axis] <= slice[1]])


# voxelise a pointset and return the corresponding tensor

def totensor(pointset, bbx, voxelsize, dtype=int):
    tensorshape = ((array(bbx[1]) - bbx[0]) / voxelsize + 1).astype(int)
    tensor = zeros(tensorshape, dtype=dtype)
    tensor[tuple(((array(pointset) - bbx[0]) / voxelsize).astype(int).T)] = 1
    return tensor


# voxelise several pointsets and return the corresponding tenors

def totensors(pointsets, bbx, voxelsize, dtype=int):
    return [totensor(ps, bbx, voxelsize, dtype=dtype) for ps in pointsets]

