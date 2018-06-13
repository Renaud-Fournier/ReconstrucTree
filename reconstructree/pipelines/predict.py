from reconstructree import *


modelpath = "/home/fournierr/Documents/Stage CIRAD/models/3d_saguaros_2_10000_32_0.1_3_3_32_adam_.0001_dice_coef_loss_['dice_coef']_50_32/model.h5"
datapath = "/home/fournierr/Documents/Stage CIRAD/data/Winter trees/X0036-1-1-WW.txt"

savepath = "/home/fournierr/Documents/Stage CIRAD/Samples/3d_appletrees"

voxelsize = .005
nbpatches = 10
patchsize = (32, 32, 32)

targetdim = 1

### Load Model

try:
    model = load_h5(modelpath)
    print("model {} loaded".format(modelpath))
except OSError:
    raise Exception("Model {} does not exist".format(modelpath))


### Prédictions

points = load_txt(datapath)
bbx = getbbx(points)

print("data: {} bbx: {}".format(shape(points), tuple(bbx)))

tensor = totensor(points, bbx, voxelsize)
print("tensor: {}".format(shape(tensor)))

patches = randpatches(tensor, nbpatches, patchsize, threshold=128)
input = toinput(patches)
print("input: {}".format(shape(input)))

predictions = model.predict(input)
print("predictions:{}".format(shape(predictions)))


### Visualisation

def showall(i):
    showpatches([input[i], predictions[i]], len(patchsize))


### Save samples

savesamples2(savepath, [input, predictions], len(patchsize), nbpatches)