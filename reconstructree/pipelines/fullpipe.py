from reconstructree import *

### Hyperparameters

# type
dimension = 3   # 2 or 3

# processing
datafile = "saguaros_2"

# patches
nbpatches = 10000
patchsize = 32
voxelsize = .1

# model
convdepth = 3
convdegree = 3
kernelsize = 32
optimizer = "adam_.0001"
loss = dice_coef_loss  # "binary_crossentropy"
metrics = [dice_coef]  # ["accuracy"]

# training
nbepochs = 50
batchsize = 32

# visualization
verbose = 2
clearprevious = True

# configuration
config = {"dimension": "{}d".format(dimension), "datafile": datafile,
          "nbpatches": nbpatches, "patchsize": patchsize, "voxelsize": voxelsize,
          "convdepth": convdepth, "convdegree": convdegree, "kernelsize": kernelsize,
          "optimizer": optimizer,
          "loss": loss if isinstance(loss, str) else loss.__name__,
          "metrics":  str([m if isinstance(m, str) else m.__name__ for m in metrics]),
          "nbepochs": nbepochs, "batchsize": batchsize}

# directories
datadir = "/home/fournierr/Documents/Stage CIRAD/data/"
modeldir = "/home/fournierr/Documents/Stage CIRAD/models/"

# data
modelname = "_".join(map(str, config.values()))
directory = "{}{}/".format(modeldir, modelname)

# checklist
if dimension not in (2, 3): raise ValueError('Invalid dimension: has to be 2 or 3')


### Model

try:

    ### Load Model

    model = load_model("{}model.h5".format(directory))
    print("\n\n--> Loading Model")
    print("\n{} loaded".format(modelname))

except OSError:     # then load data, compile and fit

    print("\nmodel {} does not exist yet".format(modelname))

    ### Data

    print("\n\n--> Loading Data\n")

    scanpatches, skelpatches, origins = getdata(datadir + datafile, dimension, voxelsize, patchsize,
                                                nbpatches)  # ,slice=(-.01, .01)
    print("\nscanpatches:{}\tskelpatches:{}".format(shape(scanpatches), shape(skelpatches)))

    input, output = toNNformat(scanpatches), toNNformat(skelpatches)
    print("\ninput:{}\toutput:{}".format(shape(input), shape(output)))

    ### Compile Model

    print("\n\n--> Compiling Model")

    model = metamodel((patchsize,) * dimension, depht=convdepth, degree=convdegree, kernel=kernelsize)
    # model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.compile(optimizer=Adam(lr=.0001), loss=loss, metrics=metrics)

    print("\nModel compiled")

    ### Fit Model

    print("\n\n--> Fitting Model", "\n" if verbose else "")

    history = fit1(model, input, output, nbepochs, batchsize, verbose)
    # history = fit2(model, input, output, nbepochs, batchsize, 32)

    print("\nhistory:{}".format(history))


### Prédictions

print("\n\n--> Making Predictions\n")

predscanpatches, predskelpatches, predorigins = getdata(datadir + datafile, dimension, voxelsize, patchsize, 10)
print("\n\nscanpatches:{}\tskelpatches:{}".format(shape(predscanpatches), shape(predskelpatches)))

predictions = model.predict(toNNformat(predscanpatches), verbose=0)
print("\npredictions:{}".format(shape(predictions)))


### Visualisation

def showall(i):
    showpatches([predscanpatches[i], predskelpatches[i], predictions[i]], dimension, clearprevious)

def showboth(i):
    showpatches([predscanpatches[i], predskelpatches[i]], dimension, clearprevious)

def showsum():
    print(model.summary())

def showhist():
    showhistory(json.load(open('{}history.json'.format(directory))) if model else history.history, clearprevious)


### Save model

print("\n\n--> Saving Model")

saveall(directory, config, model, history, predscanpatches, predskelpatches, predictions, dimension)

print("\nmodel {} saved".format(modelname))


print("\n\nProcess Terminated\n")