# Generate the samples for the banana distribution
import numpy as np
import matplotlib.pyplot as plt
import mcmc
import sys
 

# Potential for the banana distribution
def ban_U(x):
    return (1.-x[0])**2 + 10.*((x[1] - x[0]**2)**2)

def ban_gradU(x):
    return np.array([-2*(1-x[0]) - 40*(x[1] - x[0]**2)*x[0],
                            20. * (x[1] - x[0]**2)])

h_metropolis = 0.1
num_samples = 10000
skip_n_samples = 5
conv_samples = 1000
L_domain = 10
parallel = True

SAMPLING_SINGLE_CHAIN = True
SAMPLING_TO_CHECK_CONVERGENCE = True

if SAMPLING_SINGLE_CHAIN:
    X, runtime, _, _ = mcmc.chain_rwMetropolis(np.array([4, 1]), h_metropolis,
            ban_U, num_samples, skip_n_samples, L_domain)

    info_str = "INFOSIMU: chain_rwMetropolis, h = " + str(h_metropolis) + \
                " runtime: " + runtime + " n_samples = " + str(num_samples)+'\n'

    filename = "banana_chain.smp"
    if (len(sys.argv) == 2):
        filename = str(sys.argv[1]) + "_chain.smp"    
    # Store the samples into a separate file, to incentivate a chain approach
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
    X, a_rate = mcmc.convergenceMetropolis(np.array([4, 1]), h_metropolis,
        ban_U, num_samples, skip_n_samples, L_domain, conv_samples, parallel)
    info_str = "CONVERGENCE OF: chain_rwMetropolis, h = " + str(h_metropolis)+\
            " n_samples = " + str(num_samples) + " Average acceptance rate: "+\
            str(a_rate) + "%" + " skip rate: " + str(skip_n_samples) + \
            " #chains for studying convergence: " + \
            str(conv_samples) + "\n"
    
    # Store the samples into a separate file, to incentivate a chain approach
    filename = "banana_convergence.smp"
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
