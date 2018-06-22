from numpy import *
from keras.models import load_model
from reconstructree.model import customs
import types


# load a pointset from .xyz (or .txt) file (containing rows of x, y, z coords separated by spaces)

def load_xyz(path):
    return array(loadtxt(path))


# load arrays contained in a npz files (usually the arrays "scan" "skel" and "bounds")

def load_npz(path, keys=None):
    l = load(path)
    return [l[k] for k in keys] if keys else [l[k] for k in l.files]


# load a trained model from an .h5 file (with the added customs functions from model.customs.py)

def load_h5(path):
    custom_objects = {func: getattr(customs, func) for func in dir(customs) if isinstance(getattr(customs, func), types.FunctionType)}
    return load_model(path, custom_objects=custom_objects)



