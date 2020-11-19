import numpy as np
import matplotlib.pyplot as plt
import mcmc
import classif_potential
import random
import sys
import multiprocessing as mp

k_enlarge = 100
h_metropolis = 0.05
h = h_metropolis
num_samples = 50
skip_n_samples = 1
conv_samples = 1
L_domain = 10
parallel = True
#batch_size = 500

# Potential for the neural network training
def auxiliary_tot_cost(params_together):
    return classif_potential.keras_cost(params_together)

# The gradient of the potential above
def auxiliary_gradient(params_together):
    return None


# Just to make notation easier
def U(x):
    return auxiliary_tot_cost(x)

def gradU(x):
    return auxiliary_gradient(x)


SAMPLING_SINGLE_CHAIN = True
SAMPLING_TO_CHECK_CONVERGENCE = False #True

METROPOLIS_RW = False #True
ULA = False #True
MALA = False
MULTICHAIN_RW = True
# 10 chains to produce in parallel
multich = mp.cpu_count()
#multich = 10
dim = 29

if SAMPLING_SINGLE_CHAIN:
    print("Constructing a single full chain")
    if METROPOLIS_RW:
        print("...Metropolis RW")
        X, runtime, _, _ = \
            mcmc.chain_rwMetropolis(np.random.random_sample(dim),
                h, auxiliary_tot_cost, num_samples, skip_n_samples, L_domain)
        info_str = "INFOSIMU: chain_rwMetropolis, h = " + str(h) + \
            " runtime: " + runtime + " n_samples = " + str(num_samples) + '\n'
    elif MULTICHAIN_RW:
        print("...multichain RW approach")
        X, arate, _  = \
            mcmc.multichainRW(dim, L_domain, h, U, num_samples, multich,
                                                        skip_n_samples, True)
        info_str = "INFOSIMU: Multichain RW"


    # Store the samples into a separate file, modular approach
    filename = "classif_chain.smp"
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
