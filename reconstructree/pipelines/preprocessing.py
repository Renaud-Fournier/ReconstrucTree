from reconstructree.persistence.loading import *
from reconstructree.data.tensors import *
from reconstructree.data.pointsets import *
from reconstructree.data.patch import *
import os
import sys


# return neural network training input and references created from patches extracted in a directory at path containing npz files
# each npz file must contain the arrays "scan", "skel" and "bounds"

def preprocesstrain(path, nbpatches, voxelsize, patchsize):
    file_list = os.listdir(path)
    input, output = [], []
    for i in range(nbpatches):
        # print("\rpatch {}/{}".format(i+1, nbpatches), end="")
        sys.stdout.flush()
        file = "{}/{}".format(path, file_list[random.randint(0, len(file_list))])
        data = load_npz(file,["scan", "skel", "bounds"])
        scan, skel, bbx = data[0], data[1], (floor(data[2][0]), ceil(data[2][1]))
        tensors = totensors([scan, skel], bbx, voxelsize)
        patches = samerandpatch(tensors, patchsize)
        scanpatch, skelpatch = patches[0], patches[1]
        input.append(scanpatch.to_nn_format()); output.append(skelpatch.to_nn_format())
    # print("\n")
    return array(input), array(output)


# return neural network input created from patches extracted in a .txt file
# are also returned the orgins of the patches, the shape of the created tensor and the bounding box of the pointset

def preprocesspredict(path, voxelsize, patchsize):
    pointset = load_txt(path)
    bbx = getbbx(pointset)
    tensor = totensor(pointset, bbx, voxelsize)
    origins = regularorigins(shape(tensor), patchsize)
    predpatches = patches(tensor, origins, patchsize)
    input = toinput(predpatches)
    return input, origins, shape(tensor), bbx








