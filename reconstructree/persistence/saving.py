from reconstructree.visualisation.gui import *
import json
import os


# save a model at the given path (.h5 format)

def savemodel(path, model):
    model.save(path)


# save a model summary at the given path (.txt format)

def savesummary(path, model):
    with open(path, "w") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))


# save a model history at the given path (.json format)

def savehistory(path, history):
    with open(path, "w") as f:
        f.write(json.dumps(history, indent=4))


# save a picture of a plotted model history at the given path (.png format)

def savehistplot(path, history):
    ioff()
    showhistory(history, True)
    savefig(path)
    ion()


# save pictures of patches in a "samples" directory at the given path (.png format)

def savesamples(path, patches, dimension, nbsamples):
    ioff()
    for i in range(nbsamples):
        showpatches([patch[i] for patch in patches], dimension)
        savefig("{}/sample{}.png".format(path, i))
    ion()


# save a pointset at the given path (.txt format)

def savepointset(path, pointset):
    with open(path, "w") as f:
        for p in pointset: f.write(' '.join(map(str, p)) + '\n')
