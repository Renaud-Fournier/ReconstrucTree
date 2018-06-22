from reconstructree import *

# we specify the size of the patches and the voxel size

patchsize = (32, 32, 32)
voxelsize = .1

# datapath : .txt or .xyw file path containing rows of coords
# modelpath : .h5 file path containing a trained model
# savepath : .txt file path of were to save the result pointset

datapath = "/home/fournierr/Documents/Stage CIRAD/data/test trees/saguaro.xyz"
savepath = "/home/fournierr/Documents/Stage CIRAD/data/test trees/saguaro_predict.txt"
modelpath = "/home/fournierr/Documents/Stage CIRAD/models/3d_saguaros_2_10000_32_0.1_3_3_32_adam_.0001_dice_coef_loss_['dice_coef']_50_32/model.h5"


# we launch the predict function returning the prediction pointset (and also saving it to "savepath")

# 5 batches to load the data in 5 time to prevent memory errors
# verbose = 1 to print info during execution

result = predict(datapath, modelpath, savepath, voxelsize, patchsize, batches=5, verbose=1)
