# Studying the convergence for the sampling of the gaussian distribution
import numpy as np
import matplotlib.pyplot as plt
import mcmc
import sys

# Potential for a 1dim Gaussian
def U(x):
    return x**2 / 2

def gradU(x):
    return x

h_metropolis = 5
num_samples = 10000
skip_n_samples = 3
conv_samples = 100
L_domain = 10
parallel = True

# Sample candidated from HMC to minimize the potential
#X, runtime = mcmc.sample_uHMC(np.array([4]), 1, 0.1, 0.5, gradU, 10000)

X, a_rate = mcmc.convergenceMetropolis(np.random.random_sample(1),
        h_metropolis, U, num_samples, skip_n_samples, L_domain, conv_samples,
        parallel)
info_str = "CONVERGENCE OF: chain_rwMetropolis, h = " + str(h_metropolis) + \
        " n_samples = " + str(num_samples) + " Average acceptance rate: " + \
        str(a_rate) + "%" + " skip rate: " + str(skip_n_samples) + \
        " #chains for studying convergence: " + \
        str(conv_samples) + "\n"

# Store the samples into a separate file, to incentivate a C approach
filename = "gaussian_convergence_samples.smp"
if (len(sys.argv) == 2):
    filename = str(sys.argv[1]) + ".smp"
samples_file = open(filename, "w")
samples_file.write(info_str)
for x in X:
    for i in x:
        print(i, file = samples_file)
samples_file.close()
print("Samples and information stored in " + filename)
