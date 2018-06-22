from numpy import *
from reconstructree.data.pointsets import *


######    NOT USED    ######


class Sparsor:

    indices = []
    values = []
    shape = []
    default = 0

    def __init__(self, indices, values, shape, default):
        self.indices = indices
        self.values = values
        self.shape = shape
        self.default = default

    @classmethod
    def from_tensor(cls, tensor, condition=lambda v: v > 0):
        indices, values = [], []
        for i, v in ndenumerate(array(tensor)):
            if condition(v):
                indices.append(i)
                values.append(v)
        return cls(indices, values, shape(tensor), 0)

    @classmethod
    def from_pointset(cls, pointset, voxelsize, bbx=None, condition=lambda v: v > 0):
        tensor = totensor(pointset, voxelsize, bbx=bbx)
        indices, values = [], []
        for i, v in ndenumerate(array(tensor)):
            if condition(v):
                indices.append(i)
                values.append(v)
        return cls(indices, values, shape(tensor), 0)

    def getindexindex(self, index):
        return next((i for i, v in enumerate(self.indices) if all(equal(v, index))), None)

    def getval(self, index):
        ii = self.getindexindex(index)
        return self.values[ii] if ii is not None else self.default

    def setval(self, index, value):
        ii = self.getindexindex(index)
        if ii is None:
            self.indices.append(index)
            self.values.append(value)
        else:
            self.values[ii] = value
