# Generate the samples for the gaussian distribution
import numpy as np
import matplotlib.pyplot as plt
import mcmc
import sys

# Potential for a 1dim Gaussian
def U(x):
    return x**2 / 2

def gradU(x):
    return x

h_metropolis = 1
num_samples = 10000
skip_n_samples = 5
conv_samples = 500
L_domain = 10
parallel = True

SAMPLING_SINGLE_CHAIN = True
SAMPLING_TO_CHECK_CONVERGENCE = True

if SAMPLING_SINGLE_CHAIN:
    print("Sampling a single chain")
    X, runtime, _, _ = mcmc.chain_rwMetropolis(np.random.random_sample(1),
                     h_metropolis, U, num_samples, skip_n_samples)
    info_str = "INFOSIMU: chain_rwMetropolis, h = " + str(h_metropolis) + \
                " runtime: " + runtime + " n_samples = " + str(num_samples)+'\n'
    
    # Store the samples into a separate file, to incentivate a C approach
    filename = "gaussian_chain.smp"
    if (len(sys.argv) == 2):
        filename = str(sys.argv[1]) + "_chain.smp"
    samples_file = open(filename, "w")
    samples_file.write(info_str)
    for x in X:
        for i in x:
            print(i, file = samples_file)
    samples_file.close()
    print("Samples and information stored in " + filename)

if SAMPLING_TO_CHECK_CONVERGENCE:
    print("Producing multiple chains to check MC convergence")
    X, a_rate = mcmc.convergenceMetropolis(np.random.random_sample(1),
        h_metropolis, U, num_samples, skip_n_samples, L_domain, conv_samples,
        parallel)
    info_str = "CONVERGENCE OF: chain_rwMetropolis, h = " + str(h_metropolis)+ \
            " n_samples = " + str(num_samples) + " Average acceptance rate: "+ \
            str(a_rate) + "%" + " skip rate: " + str(skip_n_samples) + \
            " #chains for studying convergence: " + \
            str(conv_samples) + "\n"

    # Store the samples into a separate file, modular approach
    filename = "gaussian_convergence.smp"
    if (len(sys.argv) == 2):
        filename = str(sys.argv[1]) + "_convergence.smp"
    samples_file = open(filename, "w")
    samples_file.write(info_str)
    for x in X:
        for i in x:
            print(i, file = samples_file)
    samples_file.close()
    print("Samples and information stored in " + filename)
