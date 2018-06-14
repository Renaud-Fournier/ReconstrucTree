from reconstructree.visualisation.gui import *
import json
import os


def savemodel(directory, model):
    model.save(directory + "model.h5")


def saveconfig(directory, config):
    with open(directory + "config.json", "w") as f:
        f.write(json.dumps(config, indent=4))


def savesummary(directory, model):
    with open(directory + "summary.txt", "w") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))


def savehistory(directory, history):
    with open(directory + "history.json", "w") as f:
        f.write(json.dumps(history, indent=4))


def savehistplot(directory, history):
    ioff()
    showhistory(history, True)
    savefig(directory + 'histplot.png')
    ion()


def savesamples2(path, patches, dimension, nbsamples):
    ioff()
    for i in range(nbsamples):
        showpatches([patch[i] for patch in patches], dimension)
        savefig("{}/sample{}.png".format(path, i))
    ion()


def savesamples(directory, scanpatches, skelpatches, predictions, dimension, nbsample=10):
    ioff()
    os.makedirs(directory + "samples/", exist_ok=True)
    for i in range(nbsample):
        showpatches([scanpatches[i], skelpatches[i], predictions[i]], dimension, True)
        savefig(directory + "samples/sample" + str(i) + ".png")
    ion()


def saveall(directory, config, model, history, scanpatches, skelpatches, predictions, dimension):
    os.makedirs(directory, exist_ok=True)
    savemodel(directory, model)
    saveconfig(directory, config)
    savesummary(directory, model)
    savehistory(directory, history)
    savehistplot(directory, history)
    savesamples(directory, scanpatches, skelpatches, predictions, dimension, nbsample=10)


def savetrain(directory, config, model, history):
    os.makedirs(directory, exist_ok=True)
    savemodel(directory, model)
    saveconfig(directory, config)
    savesummary(directory, model)
    savehistory(directory, history)
    savehistplot(directory, history)


def savepointset(path, pointset):
    with open(path, "w") as f:
        for p in pointset:
            f.write(' '.join(map(str, p)) + '\n')