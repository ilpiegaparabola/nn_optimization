import numpy as np
import matplotlib.pyplot as plt
import mcmc
import nn_potential
import random
import sys

# Multiply the cost function by this constant in order to make optim. easier
k_enlarge = 100

h_metropolis = 0.1
num_samples = 50000
skip_n_samples = 4
L_domain = 20
conv_samples = 10000
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


X, a_rate = mcmc.convergenceMetropolis(np.random.random_sample(23) * 0.5,
       h_metropolis, auxiliary_tot_cost, num_samples, skip_n_samples, L_domain,
       conv_samples, parallel)
info_str = "CONVERGENCE OF: chain_rwMetropolis, h = " + str(h_metropolis) + \
        " n_samples = " + str(num_samples) + " Average acceptance rate: " + \
        str(a_rate) + "%" + " skip rate: " + str(skip_n_samples) + \
        " #chains for studying convergence: " + \
        str(conv_samples) + "\n"


# Store the samples into a separate file, to incentivate a chain approach
filename = "nn_convergence_samples.smp"
if (len(sys.argv) == 2):
    filename = str(sys.argv[1]) + ".smp"
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
