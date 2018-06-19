from numpy import *
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D


# display a MatPlotLib scatter plot for each patch in patches, all inside the same figure

def showpatches(patches, dimension, clearprevious=True):
    if clearprevious: close("all")
    f = figure(figsize=(4.5 * len(patches), 4))
    for id, patch in enumerate(patches):
        ax = f.add_subplot(1, len(patches), id + 1, projection='rectilinear' if dimension == 2 else '3d')
        data = array([(*i[:dimension], v) for i, v in ndenumerate(patch)])
        ax.scatter(*[data[:, d] for d in range(dimension)], s=data[:, dimension] * 10)
    if isinteractive(): show()


# display plotted curves of the history keys (error, precision ...)

def showhistory(hist, clearprevious=True):
    if clearprevious: close("all")
    figure().gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    for k in hist: plot(hist[k])
    legend(hist.keys(), loc='best')
    title('Model accuracy and loss')
    ylabel('accuracy / loss')
    xlabel('epoch')
    if isinteractive(): show()