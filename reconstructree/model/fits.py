

# alternative fit method for too large to be loaded at once data set

def alt_fit(model, input, output, nbepochs, batchsize, splitsize):
    history = {}
    for e in range(nbepochs):
        for i in range(0, len(input), splitsize):
            j = i + min(splitsize, len(input))
            hist = model.fit(input[i:j], output[i:j], batch_size=batchsize, epochs=1, verbose=0, shuffle=True, validation_split=.1)
            for k, v in hist.history.items():
                if k in history: history[k] += v
                else: history.setdefault(k, v)
            print("\repoch {} slice {} : {}\t{}".format(e, i, j, hist.history), end="")
    return history
