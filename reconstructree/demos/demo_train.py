from reconstructree import *

patchsize = (32, 32, 32)
voxelsize = 0.1
nbpatches = 10000

input, output = preprocesstrain("/home/fournierr/Documents/Stage CIRAD/data/saguaros_2", nbpatches, voxelsize, patchsize)

model = metamodel(patchsize, depht=3, degree=3)
model.compile(optimizer=Adam(lr=.0001), loss=dice_coef_loss, metrics=[dice_coef])
model.summary()

history = model.fit(input, output, batch_size=32, epochs=50, verbose=2, shuffle=True, validation_split=.1)
showhistory(history.history)

model.save("/home/fournierr/Documents/Stage CIRAD/models/prettymodel")
