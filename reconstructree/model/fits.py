

def fit1(model, input, output, nbepochs, batchsize, verbose=2):
    hist = model.fit(input, output, batch_size=batchsize, epochs=nbepochs, verbose=verbose, shuffle=True, validation_split=.1)
    return hist


def fit2(model, input, output, nbepochs, batchsize, splitsize):
    history = {}
    for e in range(nbepochs):
        print("\n")
        for i in range(0, len(input), splitsize):
            j = i + min(splitsize, len(input))
            hist = model.fit(input[i:j], output[i:j], batch_size=batchsize, epochs=1, verbose=0, shuffle=True,
                             validation_split=.1)
            for k, v in hist.history.items():
                if k in history: history[k] += v
                else: history.setdefault(k, v)
            # print("epoch " + str(e) + " slice " + str(i) + ":" + str(j) + "\t" + str(hist.history))
            print("\repoch " + str(e) + " slice " + str(i) + ":" + str(j) + "\t" + str(hist.history), end="")
    return history