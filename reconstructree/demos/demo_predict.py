from reconstructree import *

patchsize = (32, 32, 32)
voxelsize = .003

input, origins, tensorshape, bbx = preprocesspredict("/home/fournierr/Documents/Stage CIRAD/data/Winter trees/X0036-1-1-WW.txt", voxelsize, patchsize)

model = load_h5("/home/fournierr/Documents/Stage CIRAD/models/3d_saguaros_2_10000_32_0.1_3_3_32_adam_.0001_dice_coef_loss_['dice_coef']_50_32/model.h5")

predictions = model.predict(input)

# showpatches([input[0], predictions[0]], 3)

predpointset = postprocesspredict(predictions, origins, tensorshape, voxelsize, bbx)

savepointset("/home/fournierr/Documents/Stage CIRAD/data/testtree.txt", predpointset)