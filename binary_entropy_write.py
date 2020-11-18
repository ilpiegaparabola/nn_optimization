import numpy as np
import rw
import nnlib
import random
import sys
import multiprocessing as mp

h = 1
nsamples = 10
thin = 1
#conv_samples = 1
L = 10
nchains = 6

d = nnlib.global_NNDIMENSION

# Just to make notation easier
def U(x):
    return nnlib.loss(x)


SAMPLING_SINGLE_CHAIN = True
SAMPLING_TO_CHECK_CONVERGENCE = False #True


if SAMPLING_SINGLE_CHAIN:
    print("Constructing a single full chain")
    print("...multichain RW approach")
    X, arate, _  = rw.multiRW(d, h, U, nsamples, nchains, thin, L)
    info_str = "INFOSIMU: Multichain RW"


    # Store the samples into a separate file, modular approach
    filename = "bninary_chain.smp"
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
