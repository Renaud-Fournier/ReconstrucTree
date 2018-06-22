from reconstructree.persistence.loading import *
from reconstructree.data.pointsets import *
from reconstructree.persistence.saving import *
from numpy import *


# make predictions on poinset extracted from the "datapath" file (.xyz or .txt) with the model loaded at "modelpath"
# return the corresponding prediction pointset and save it to "savepath"
# verbose to 1 for printed info during execution


def predict(datapath, modelpath, savepath, voxelsize, patchsize, batches=1, verbose=0):

    if verbose: print("\ninpointset", end=' ')
    inpointset = load_xyz(datapath)  # load pointset
    if verbose: print(shape(inpointset))

    bbx = getbbx(inpointset)

    if verbose: print("\nintensor", end=' ')
    intensor = Tensor.from_pointset(inpointset, voxelsize, bbx)  # create input Tensor
    if verbose: print(intensor.shape)

    if verbose: print("\norigins", end=' ')
    origins = intensor.regular_origins(patchsize)  # extract regular origins
    if verbose: print(shape(origins))

    if verbose: print("\norigins_slices", end=' ')
    origins_slices = array_split(origins, batches)  # separate the origins in "batches" different groups
    if verbose: print(shape(origins_slices))

    if verbose: print("\nmodel", end=' ')
    model = load_h5(modelpath)  # load the model
    if verbose: print("loaded")

    if verbose: print("\nouttensor", end=' ')
    outtensor = Tensor(zeros(shape(intensor)), shape(intensor), voxelsize, bbx)  # create empty output Tensor
    if verbose: print(outtensor.shape)

    c = 0

    for origins in origins_slices:  # we load only a part of the data set at once to prevent memory errors

        c += 1
        if verbose: print("\nbatch {}".format(c))

        if verbose: print("inpatches", end=' ')
        inpatches = intensor.get_patches(origins, patchsize)  # extract the Patches
        if verbose: print(shape(inpatches))

        if verbose: print("input", end=' ')
        input = toinput(inpatches)  # convert Patches to neural network input
        if verbose: print(shape(input))

        if verbose: print("predictions", end=' ')
        output = model.predict(input, verbose=verbose)  # make predictions
        if verbose: print("output", end=' ')
        if verbose: print(shape(output))

        if verbose: print("outpatches", end=' ')
        outpatches = patches_from_nn_format(output, origins)  # create output Patches from neural network output
        if verbose: print(shape(outpatches))

        if verbose: print("merge_patches")
        outtensor.merge_patches(outpatches)  # insert the Patches in the output Tensor

    if verbose: print("\noutpointset", end=' ')
    outpointset = outtensor.to_pointset(threshold=.1)  # create output pointset from Tensor
    if verbose: print(shape(outpointset))

    savepointset(savepath, outpointset)  # save output pointset to "savepath"

    return outpointset
