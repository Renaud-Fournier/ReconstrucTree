from reconstructree.data.pointsets import *
from reconstructree.persistence.loading import *


# return neural network training input extracted fro directory of .npz files (containing "scan", "skel" and "bounds" arrays)

def preprocess_train_data(path, nbpatches, voxelsize, patchsize, verbose=0):

    file_list = os.listdir(path)
    input, output = [], []

    for i in range(nbpatches):

        if verbose: print("\rpatch {}/{}".format(i+1, nbpatches), end="")
        sys.stdout.flush()

        file = "{}/{}".format(path, file_list[random.randint(0, len(file_list))])
        data = load_npz(file, ["scan", "skel", "bounds"])

        scan, skel, bbx = data[0], data[1], (floor(data[2][0]), ceil(data[2][1]))

        tensors = to_tensors([scan, skel], bbx, voxelsize)

        patches = get_same_random_patch(tensors, patchsize)

        scanpatch, skelpatch = patches[0], patches[1]

        input.append(scanpatch.to_nn_format())
        output.append(skelpatch.to_nn_format())

    if verbose: print("\n")

    return array(input), array(output)
