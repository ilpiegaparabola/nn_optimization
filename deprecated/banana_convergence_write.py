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
num_samples = 20000
skip_n_samples = 3
L_domain = 10
conv_samples = 500
parallel = True

# Sample candidated from HMC to minimize the potential
#X, runtime = mcmc.sample_uHMC(np.array([4,1]),0.5,0.001, 0.5, ban_gradU, 10000)

X, a_rate = mcmc.convergenceMetropolis(np.array([4, 1]), h_metropolis,
        ban_U, num_samples, skip_n_samples, L_domain, conv_samples, parallel)
info_str = "CONVERGENCE OF: chain_rwMetropolis, h = " + str(h_metropolis) + \
        " n_samples = " + str(num_samples) + " Average acceptance rate: " + \
        str(a_rate) + "%" + " skip rate: " + str(skip_n_samples) + \
        " #chains for studying convergence: " + \
        str(conv_samples) + "\n"

# Store the samples into a separate file, to incentivate a chain approach
filename = "banana_convergence_samples.smp"
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
