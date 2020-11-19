import numpy as np
import rw
import nnlib
import random
import sys
import multiprocessing as mp

#nnlib.plot_data()

h = 0.1
nsamples = 5000
thin = 5
#conv_samples = 1
L = 5
nchains = 12

d = nnlib.get_num_params()

# Just to make notation easier
def U(x):
    return nnlib.loss(x)


SAMPLING_SINGLE_CHAIN = True
SAMPLING_TO_CHECK_CONVERGENCE = False #True

SIMPLE_RW = 1 #True # When false, performs the more efficient multichain

if SAMPLING_SINGLE_CHAIN:
    print("Constructing a single full chain")
    if SIMPLE_RW:
        print("...simple single RW (DEBUG)")
        input("")
        startx = np.random.uniform(-L, L, d)
        X, info_str, arate, _ = rw.chainRW(startx, h, U, nsamples, thin, 
                                                                L, verbose = 2)
        print("Classifiation accuracy using the last sample: ",
                nnlib.accuracy_with_l2err(X[-1]))
    else:
        print("...multichain RW approach")
        input("")
        X, arate, _  = rw.multiRW(d, h, U, nsamples, nchains, thin, L)
        info_str = "INFOSIMU: Multichain RW"
        print("Classifiation accuracy using the last sample: ",
                nnlib.accuracy_with_l2err(X[-1]))


    # Store the samples into a separate file, modular approach
    filename = "binary_chain.smp"
    if (len(sys.argv) == 2):
        filename = str(sys.argv[1]) + "_chain.smp"
    samples_file = open(filename, "w")
    samples_file.write(info_str)
    for x in X:
        for i in range(len(x)):
            if i < (len(x) - 1):
                print(x[i], file = samples_file, end = ' ')
            else:
                print(x[i], file = samples_file, end = '\n')
    samples_file.close()
    print("Samples and information stored in " + filename)
