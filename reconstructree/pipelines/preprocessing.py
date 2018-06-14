from reconstructree.persistence.loading import *
from reconstructree.data.tensors import *
from reconstructree.data.pointsets import *
import os


def preprocess(path, nbpatches, voxelsize, patchsize):
    file_list = os.listdir(path)
    input, output = [], []
    for i in range(nbpatches):
        file = "{}/{}".format(path, file_list[random.randint(0, len(file_list))])
        data = load_npz(file,["scan", "skel", "bounds"])
        scan, skel, bbx = data[0], data[1], data[2]
        #print(shape(scan), shape(skel), bbx)
        tensors = totensors([scan, skel], bbx, voxelsize)
        tensorscan, tensorskel = tensors[0], tensors[1]
        #print(shape(tensorscan), shape(tensorskel))
        patches = samerandpatch(tensors, patchsize)
        scanpatch, skelpatch = patches[0], patches[1]
        #print(shape(scanpatch.to_nn_format()), shape(skelpatch.to_nn_format()))
        input.append(scanpatch.to_nn_format()); output.append(skelpatch.to_nn_format())
    return array(input), array(output)








