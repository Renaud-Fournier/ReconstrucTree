from reconstructree import *

model = load_h5("/home/fournierr/Documents/Stage CIRAD/models/3d_saguaros_2_10000_32_0.1_3_3_32_adam_.0001_dice_coef_loss_['dice_coef']_50_32/model.h5")
pointset = load_txt("/home/fournierr/Documents/Stage CIRAD/data/Winter trees/X0036-1-1-WW.txt")

bbx = getbbx(pointset)
print(shape(pointset))

voxelsize = .003
patchsize = (32, 32, 32)

tensor = totensor(pointset, bbx, voxelsize)
tensorshape = shape(tensor)
print(tensorshape)

origins = regularorigins(tensorshape, patchsize)
print(shape(origins))

patches = patches(tensor, origins, patchsize)
input = toinput(patches)
print(shape(input))

predictions = model.predict(input, verbose=1)
print(shape(predictions))

# showpatches([input[0], predictions[0]], 3)

predpatches = patches_from_nn_format(predictions, origins)
print(shape(predpatches))

predtensor = fast_tensor_from_patches(predpatches, tensorshape)
print(shape(predtensor))

predpointset = topointset(predtensor, voxelsize, bbx, threshold=.05)
print(shape(predpointset))


savepointset("/home/fournierr/Documents/Stage CIRAD/data/testtree.txt", predpointset)