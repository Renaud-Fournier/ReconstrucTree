from reconstructree import *


### Load Model

modelpath = "/home/fournierr/Documents/Stage CIRAD/models/3d_saguaros_2_10000_32_0.1_3_3_32_adam_.0001_dice_coef_loss_['dice_coef']_50_32/model.h5"

try:
    model = load_h5(modelpath)
    print("model {} loaded".format(modelpath))
except OSError:
    raise Exception("Model {} does not exist".format(modelpath))


### Preprocessing

datapath = "/home/fournierr/Documents/Stage CIRAD/data/Winter trees/X0036-1-1-WW.txt"

pointset = load_txt(datapath)
bbx = getbbx(pointset)
print("data: {} bbx: {}".format(shape(pointset), tuple(bbx)))

voxelsize = .01

tensor = totensor(pointset, bbx, voxelsize)
tensorshape = shape(tensor)
print("tensor: {}".format(tensorshape))

patchsize = (32, 32, 32)

origins = regularorigins(tensorshape, patchsize)

patches = patches(tensor, origins, patchsize)
input = toinput(patches)
print("input: {}".format(shape(input)))


### Prédictions

predictions = model.predict(input)
print("predictions:{}".format(shape(predictions)))


### Postprocessing

predpatches = patches_from_nn_format(predictions, origins)
print("predpatches:{}".format(shape(predpatches)))

predtensor = tensor_from_patches(predpatches, tensorshape)
print("predtensor:{}".format(shape(predtensor)))

predpointset = topointset(predtensor, voxelsize, bbx, threshold=.05)
print("predpointset:{}".format(shape(predpointset)))


### Save samples

savepath = "/home/fournierr/Documents/Stage CIRAD/data/testtree.txt"

savepointset(savepath, predpointset)

