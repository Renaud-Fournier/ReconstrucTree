from reconstructree import *

### Hyperparameters

# directories
modeldir = "/home/fournierr/Documents/Stage CIRAD/models"
datapath = "/home/fournierr/Documents/Stage CIRAD/data/pretty trees"

# type
dimension = 3   # 2 or 3

# patches
nbpatches = 1
patchsize = (32, 32, 32)
voxelsize = .1

# model
convdepth = 1
convdegree = 1
kernelsize = 32
optimizer = "adam_.0001"
loss = dice_coef_loss  # "binary_crossentropy"
metrics = [dice_coef]  # ["accuracy"]

# training
nbepochs = 1
batchsize = 32

# visualization
verbose = 2
clearprevious = True

# configuration
config = {"dimension": "{}d".format(dimension), "datapath": datapath,
          "nbpatches": nbpatches, "patchsize": patchsize, "voxelsize": voxelsize,
          "convdepth": convdepth, "convdegree": convdegree, "kernelsize": kernelsize,
          "optimizer": optimizer,
          "loss": loss if isinstance(loss, str) else loss.__name__,
          "metrics":  str([m if isinstance(m, str) else m.__name__ for m in metrics]),
          "nbepochs": nbepochs, "batchsize": batchsize}


### Data

print("\n\n--> Preprocessing\n")

input, output = preprocess(datapath, nbpatches, voxelsize, patchsize)

print(shape(input), shape(output))

### Compile Model

print("\n\n--> Compiling Model")

#model = metamodel(patchsize, depht=convdepth, degree=convdegree, kernel=kernelsize)

model = examplemodel(patchsize)

model.compile(optimizer=Adam(lr=.0001), loss=loss, metrics=metrics)

print("\nModel compiled")

### Fit Model

print("\n\n--> Fitting Model", "\n" if verbose else "")

history = fit1(model, input, output, nbepochs, batchsize, verbose)

print("\nhistory:{}".format(history))


### Save model

#print("\n\n--> Saving Model")

#savetrain(modeldir + "/test", config, model, history)

#print("\nmodel {} saved".format(modeldir + "/test"))


print("\n\nProcess Terminated\n")