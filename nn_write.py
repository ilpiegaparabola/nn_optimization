import numpy as np
import matplotlib.pyplot as plt
import mcmc
import nn_potential
import random
import sys

k_enlarge = 100
h_metropolis = 0.2
h = h_metropolis
num_samples = 10000
skip_n_samples = 5
conv_samples = 1000
L_domain = 10
parallel = True

# Potential for the neural network training
def auxiliary_tot_cost(params_together):
    # here x, y are understood as global variables defined above
    b2, b3, b4, W2, W3, W4 = nn_potential.split_params(params_together)
    return nn_potential.total_cost (nn_potential.x, 
            nn_potential.y, b2, b3, b4, W2, W3, W4, 10) * k_enlarge


# The gradient of the potential above
def auxiliary_gradient(params_together):
    b2, b3, b4, W2, W3, W4 = nn_potential.split_params(params_together)
    return nn_potential.grad_cost (nn_potential.x,
            nn_potential.y, b2, b3, b4, W2, W3, W4) * k_enlarge

# Just to make notation easier
def U(x):
    return auxiliary_tot_cost(x)

def gradU(x):
    return auxiliary_gradient(x)



SAMPLING_SINGLE_CHAIN = True
SAMPLING_TO_CHECK_CONVERGENCE = True

METROPOLIS_RW = False
ULA = True
MALA = False

if SAMPLING_SINGLE_CHAIN:
    print("Constructing a single full chain")
    if METROPOLIS_RW:
#        X, runtime, _, _ = \
#            mcmc.chain_rwMetropolis(np.random.random_sample(23),
#                h, auxiliary_tot_cost, num_samples, skip_n_samples, L_domain)
        X, runtime, _, _ = mcmc.chain_rwMetropolis(np.random.random_sample(23),
                                h, U, num_samples, skip_n_samples, L_domain)
        info_str = "INFOSIMU: chain_rwMetropolis, h = " + str(h) + \
            " runtime: " + runtime + " n_samples = " + str(num_samples) + '\n'
    elif ULA:
        X, runtime, _ = mcmc.ulaChain(np.random.random_sample(23),
                            h, U, gradU, num_samples, skip_n_samples, L_domain)
        info_str = "INFOSIMU: ULA, h = " + str(h) + \
                " runtime: " + runtime + " n_samples = " + str(num_samples)+'\n'
    elif MALA:
        X, runtime, _, _ = mcmc.malaChain(np.random.random_sample(23), h, U, 
                                gradU, num_samples, skip_n_samples, L_domain)
        info_str = "INFOSIMU: MALA, h = " + str(h) + \
            " runtime: " + runtime + " n_samples = " + str(num_samples) + '\n'

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
        X, a_rate = mcmc.convergenceMetropolis(np.random.random_sample(23),
                h, auxiliary_tot_cost, num_samples, skip_n_samples, L_domain,
                   conv_samples, parallel)
        info_str = "CONVERGENCE OF: chain_rwMetropolis, h = " +str(h) +\
        " n_samples = " + str(num_samples) + " Average acceptance rate: " +\
        str(a_rate) + "%" + " skip rate: " + str(skip_n_samples) +\
        " #chains for studying convergence: " +\
        str(conv_samples) + "\n"
    elif ULA:
        X = mcmc.ulaConvergence(np.random.random_sample(23), h, U, gradU,
                num_samples, skip_n_samples, L_domain, conv_samples, parallel)
        info_str = "CONVERGENCE OF: ULA, h = " + str(h)+ \
            " n_samples = " + str(num_samples) + "%" + " skip rate: " + \
            str(skip_n_samples) + \
            " #chains for studying convergence: " + \
            str(conv_samples) + "\n"
    elif MALA:
        X, a_rate = mcmc.malaConvergence(np.random.random_sample(23), h, U,
                                 gradU, num_samples, skip_n_samples, L_domain,
                                                        conv_samples, parallel)
        info_str = "CONVERGENCE OF: MALA, h = " +str(h) +\
        " n_samples = " + str(num_samples) + " Average acceptance rate: " +\
        str(a_rate) + "%" + " skip rate: " + str(skip_n_samples) +\
        " #chains for studying convergence: " +\
        str(conv_samples) + "\n"
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
