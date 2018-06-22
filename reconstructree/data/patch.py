from numpy import *


# class Patch : associate a "value" array to a position in a tensor

class Patch:

    def __init__(self, values, origin):
        self.values = array(values)  # multi-dim array containing the patch values
        self.origin = array(origin)  # lower left corner position of the patch in its origin tensor

    @classmethod
    def from_tensor(cls, tensor, origin, patchsize):

        # create a Patch of size "patchsize" by extracting the values in a Tensor object
        # (padded with 0 if the tensor is not big enough or if the origin is outside the tensor

        patch = array(tensor.values[[slice(origin[i], origin[i] + patchsize[i]) for i, v in enumerate(tensor.shape)]])
        values = pad(patch, [(0, patchsize[i] - d) for i, d in enumerate(shape(patch))], 'constant', constant_values=0)
        return cls(values, origin)

    @classmethod
    def from_values(cls, originvalues, origin, patchsize):

        # create a Patch of size "patchsize" by extracting the values in an multi-dim array

        patch = array(originvalues[[slice(origin[i], origin[i] + patchsize[i]) for i, v in enumerate(shape(originvalues))]])
        values = pad(patch, [(0, patchsize[i] - d) for i, d in enumerate(shape(patch))], 'constant', constant_values=0)
        return cls(values, origin)

    @classmethod
    def from_nn_format(cls, output, origin):

        # create a Patch from the output of the neural network. Origin has to be specified

        values = squeeze(output, axis=len(shape(output)) - 1)
        return cls(values, origin)

    def to_nn_format(self):

        # convert the Patch to neural network input

        return self.values[..., newaxis].astype(float)


def toinput(patches):

    # convert an array of Patches to neural network input

    return array([patch.to_nn_format() for patch in patches])


def patches_from_nn_format(outputs, origins):

    # convert output of neural network to Patches

    return array([Patch.from_nn_format(v, origins[i]) for i, v in enumerate(outputs)])
