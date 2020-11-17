import numpy as np
import matplotlib.pyplot as plt
import mcmc
import mnist_potential
import random
import sys
import multiprocessing as mp

k_enlarge = 1
h_metropolis = 500
h = h_metropolis
num_samples = 10
skip_n_samples = 5
conv_samples = 1
L_domain = 2000
parallel = True
batch_size = 500

# Potential for the neural network training
def auxiliary_tot_cost(params_together):
    return mnist_potential.keras_cost(params_together)
    
# The gradient of the potential above
def auxiliary_gradient(params_together):
    return None


# Just to make notation easier
def U(x):
    return auxiliary_tot_cost(x)

def gradU(x):
    return auxiliary_gradient(x)


SAMPLING_SINGLE_CHAIN = True
SAMPLING_TO_CHECK_CONVERGENCE = True

METROPOLIS_RW = True
ULA = False #True
MALA = False
MULTICHAIN_RW = False #True
# 10 chains to produce in parallel
multich = mp.cpu_count()
#multich = 10
dim = 231926


if SAMPLING_SINGLE_CHAIN:
    print("Constructing a single full chain")
    if METROPOLIS_RW:
        print("...Metropolis RW")
#        X, runtime, _, _ = \
#            mcmc.chain_rwMetropolis(np.random.random_sample(23),
#                h, auxiliary_tot_cost, num_samples, skip_n_samples, L_domain)
        store_chain = True #False # Just want to compute expectation
        X, runtime, _, _ = mcmc.chain_rwBatchMetropolis(
                             np.random.random_sample(dim),
                             h, U, num_samples, batch_size, 
                             store_chain, skip_n_samples, L_domain)
        info_str = "INFOSIMU: chain_rwMetropolis, h = " + str(h) + \
            " runtime: " + runtime + " n_samples = " + str(num_samples) + '\n'
    elif ULA:
        print("...ULA")
        X, runtime, _ = mcmc.ulaChain(np.random.random_sample(dim),
                            h, U, gradU, num_samples, skip_n_samples, L_domain)
        info_str = "INFOSIMU: ULA, h = " + str(h) + \
                " runtime: " + runtime + " n_samples = " + str(num_samples)+'\n'
    elif MALA:
        print("...MALA")
        X, runtime, _, _ = mcmc.malaChain(np.random.random_sample(dim), h, U, 
                                gradU, num_samples, skip_n_samples, L_domain)
        info_str = "INFOSIMU: MALA, h = " + str(h) + \
            " runtime: " + runtime + " n_samples = " + str(num_samples) + '\n'
    elif MULTICHAIN_RW:
        print("...multichain RW approach")
        X, arate, _  = \
            mcmc.multichainRW(dim, L_domain, h, U, num_samples, multich,
                                                        skip_n_samples, True)
        info_str = "INFOSIMU: Multichain RW"

    # Store the samples into a separate file, modular approach
    filename = "nn_chain.smp"
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


if SAMPLING_TO_CHECK_CONVERGENCE:
    print("Producing multiple chains to check MC convergence")
    if METROPOLIS_RW:
        X, a_rate = mcmc.convergenceMetropolis(np.random.random_sample(dim),
                h, auxiliary_tot_cost, num_samples, skip_n_samples, L_domain,
                   conv_samples, parallel)
        info_str = "CONVERGENCE OF: chain_rwMetropolis, h = " +str(h) +\
        " n_samples = " + str(num_samples) + " Average acceptance rate: " +\
        str(a_rate) + "%" + " skip rate: " + str(skip_n_samples) +\
        " #chains for studying convergence: " +\
        str(conv_samples) + "\n"
    elif ULA:
        X = mcmc.ulaConvergence(np.random.random_sample(dim), h, U, gradU,
                num_samples, skip_n_samples, L_domain, conv_samples, parallel)
        info_str = "CONVERGENCE OF: ULA, h = " + str(h)+ \
            " n_samples = " + str(num_samples) + "%" + " skip rate: " + \
            str(skip_n_samples) + \
            " #chains for studying convergence: " + \
            str(conv_samples) + "\n"
    elif MALA:
        X, a_rate = mcmc.malaConvergence(np.random.random_sample(dim), h, U,
                                 gradU, num_samples, skip_n_samples, L_domain,
                                                        conv_samples, parallel)
        info_str = "CONVERGENCE OF: MALA, h = " +str(h) +\
        " n_samples = " + str(num_samples) + " Average acceptance rate: " +\
        str(a_rate) + "%" + " skip rate: " + str(skip_n_samples) +\
        " #chains for studying convergence: " +\
        str(conv_samples) + "\n"
    elif MULTICHAIN_RW:
        X = mcmc.multichainRWconvergence(dim, L_domain, h, U, num_samples, 
                multich, skip_n_samples, conv_samples)
        info_str = "CONVERGENCE of: Multichain RW"


    # Store the samples into a separate file, modular approach
    filename = "nn_convergence.smp"
    if (len(sys.argv) == 2):
        filename = str(sys.argv[1]) + "_convergence.smp"
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
