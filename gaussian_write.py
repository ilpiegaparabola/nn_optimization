# Generate the samples for the gaussian distribution
import numpy as np
import matplotlib.pyplot as plt
import mcmc
import sys
import multiprocessing as mp

# Potential for a 1dim Gaussian
def U(x):
    return x**2 / 2

def gradU(x):
    return x

h_metropolis = 0.2
h = h_metropolis
num_samples = 100
skip_n_samples = 5
conv_samples = 5
L_domain = 10
L = L_domain
parallel = True

SAMPLING_SINGLE_CHAIN = True
SAMPLING_TO_CHECK_CONVERGENCE = True

METROPOLIS_RW = False
ULA = False#True
MALA = False
MULTICHAIN_RW = True
# 10 chains to produce in parallel
#multich = 10
multich = mp.cpu_count()

if SAMPLING_SINGLE_CHAIN:
    print("Sampling a single chain")
    if METROPOLIS_RW:
        print("...Metropolis RW")
        X, runtime, _, _ = mcmc.chain_rwMetropolis(np.random.random_sample(1),
                     h_metropolis, U, num_samples, skip_n_samples)
        info_str = "INFOSIMU: chain_rwMetropolis, h = " + str(h_metropolis) + \
                " runtime: " + runtime + " n_samples = " + str(num_samples)+'\n'
    elif ULA:
        print("...ULA")
        X, runtime, _ = mcmc.ulaChain(np.random.random_sample(1),
                h, U, gradU, num_samples, skip_n_samples, L_domain)
        info_str = "INFOSIMU: ULA, h = " + str(h) + \
                " runtime: " + runtime + " n_samples = " + str(num_samples)+'\n'
    elif MALA:
        print("...MALA")
        X, runtime, _, _ = mcmc.malaChain(np.random.random_sample(1), h,
                                U, gradU, num_samples, skip_n_samples, L_domain)
        info_str = "INFOSIMU: MALA, h = " + str(h) + \
                " runtime: " + runtime + " n_samples = " + str(num_samples)+'\n'
    elif MULTICHAIN_RW:
        print("...multichain RW approach")
        X, arate, _  = \
            mcmc.multichainRW(1, L, h, U, num_samples, multich, 
                    skip_n_samples, True)
        info_str = "INFOSIMU: Multichain RW"
   
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

    if METROPOLIS_RW:
        X, a_rate = mcmc.convergenceMetropolis(np.random.random_sample(1),
             h, U, num_samples, skip_n_samples, L_domain, conv_samples,parallel)
        info_str = "CONVERGENCE OF: chain_rwMetropolis, h = " + str(h)+ \
            " n_samples = " + str(num_samples) + " Average acceptance rate: "+ \
            str(a_rate) + "%" + " skip rate: " + str(skip_n_samples) + \
            " #chains for studying convergence: " + \
            str(conv_samples) + "\n"
    elif ULA:
        X = mcmc.ulaConvergence(np.random.random_sample(1), h, U, gradU,
                num_samples, skip_n_samples, L_domain, conv_samples, parallel)
        info_str = "CONVERGENCE OF: ULA, h = " + str(h)+ \
            " n_samples = " + str(num_samples) + "%" + " skip rate: " + \
            str(skip_n_samples) + \
            " #chains for studying convergence: " + \
            str(conv_samples) + "\n"
    elif MALA:
        print("...MALA")
        X, a_rate = mcmc.malaConvergence(np.random.random_sample(1),h, U, gradU,
                num_samples, skip_n_samples, L_domain, conv_samples, parallel)
        info_str = "CONVERGENCE OF: MALA, h = " + str(h)+\
            " n_samples = " + str(num_samples) + " Average acceptance rate: "+\
            str(a_rate) + "%" + " skip rate: " + str(skip_n_samples) + \
            " #chains for studying convergence: " + \
            str(conv_samples) + "\n"
    elif MULTICHAIN_RW:
        X = mcmc.multichainRWconvergence(1, L, h, U, num_samples, multich, 
                skip_n_samples, conv_samples)
        info_str = "CONVERGENCE of: Multichain RW\n"

    # Store the samples into a separate file, modular approach
    filename = "gaussian_convergence.smp"
    if (len(sys.argv) == 2):
        filename = str(sys.argv[1]) + "_convergence.smp"
    samples_file = open(filename, "w")
    samples_file.write(info_str)
    for x in X:
  #      for i in x:
            print(x, file = samples_file)
    samples_file.close()
    print("Samples and information stored in " + filename)
