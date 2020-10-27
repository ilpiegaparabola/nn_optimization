# Generate the samples for the gaussian distribution
import numpy as np
import matplotlib.pyplot as plt
import mcmc

# Potential for a 1dim Gaussian
def U(x):
    return x**2 / 2

def gradU(x):
    return x

h_metropolis = 5
num_samples = 50000
skip_n_samples = 5

#Ignore the further results of the chain, useful only for convergence analysis
X, runtime, _, _ = mcmc.chain_rwMetropolis(np.random.random_sample(1),
                 h_metropolis, U, num_samples, skip_n_samples)
info_str = "INFOSIMU: chain_rwMetropolis, h = " + str(h_metropolis) + \
            " runtime: " + runtime + " n_samples = " + str(num_samples) + '\n'

# Store the samples into a separate file, to incentivate a C approach
filename = "gaussian_stored_samples.smp"
samples_file = open(filename, "w")
samples_file.write(info_str)
for x in X:
    for i in x:
        print(i, file = samples_file)
samples_file.close()
print("Samples and information stored in " + filename)
