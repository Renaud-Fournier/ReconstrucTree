from reconstructree import *

# we specify the number of patches to extract, their size and the voxel size

patchsize = (32, 32, 32)
voxelsize = 0.1
nbpatches = 10000

# we launch the preprocess_train_data function to generate training input and ref for the neural network

input, ref = preprocess_train_data("/home/fournierr/Documents/Stage CIRAD/data/saguaros_2", nbpatches, voxelsize, patchsize)

# we create and compile the neural network

model = enco_deco_model(patchsize, depth=3, degree=3)
model.compile(optimizer=Adam(lr=.0001), loss=dice_coef_loss, metrics=[dice_coef])

model.summary()  # to print a network summary

# we launch the fit function to begin the training
# batch_size : number of data loaded at once on the gpu
# epochs : number of training passes on the entire dataset ("duration" of the training)
# verbose=2 to print info during execution

history = model.fit(input, ref, batch_size=32, epochs=50, verbose=2, shuffle=True, validation_split=.1)
# the model is then trained, and the returned history object contains training info

# its attribute "history" can be plotted to see performance evolution
showhistory(history.history)

# we can save the trained model with model.save
model.save("/home/fournierr/Documents/Stage CIRAD/models/prettymodel")
