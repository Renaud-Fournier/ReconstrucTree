from numpy import *
from keras.models import load_model
from reconstructree.model import customs
import types


# load a pointset from txt file (rows of x, y, z coordinates separated by spaces)

def load_txt(path):
    return array(loadtxt(path))


# load arrays contained in a npz files

def load_npz(path, keys=None):
    l = load(path)
    return [l[k] for k in keys] if keys else [l[k] for k in l.files]


# load a trained model from h5 file

def load_h5(path):
    custom_objects = {func: getattr(customs, func) for func in dir(customs) if isinstance(getattr(customs, func), types.FunctionType)}
    return load_model(path, custom_objects=custom_objects)



