from reconstructree import *

patchsize = (32, 32, 32)

input, output = preprocess("/home/fournierr/Documents/Stage CIRAD/data/saguaros_2", 10000, 0.1, patchsize)

print(shape(input), shape(output))

model = metamodel(patchsize, depht=3, degree=3)

model.compile(optimizer=Adam(lr=.0001), loss=dice_coef_loss, metrics=[dice_coef])

model.summary()

# history = fit1(model, input, output, 50, 32)
history = model.fit(input, output, batch_size=32, epochs=50, verbose=2, shuffle=True, validation_split=.1)

showhistory(history.history)

model.save("/home/fournierr/Documents/Stage CIRAD/models/prettymodel")
