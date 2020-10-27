# Generate the samples for the banana distribution
import numpy as np
import matplotlib.pyplot as plt
import mcmc
 

# Potential for the banana distribution
def ban_U(x):
    return (1.-x[0])**2 + 10.*((x[1] - x[0]**2)**2)

def ban_gradU(x):
    return np.array([-2*(1-x[0]) - 40*(x[1] - x[0]**2)*x[0],
                            20. * (x[1] - x[0]**2)])

h_metropolis = 0.1
num_samples = 250000
skip_n_samples = 1


# Sample candidated from HMC to minimize the potential
#X, runtime = mcmc.sample_uHMC(np.array([4,1]),0.5,0.001, 0.5, ban_gradU, 10000)
X, runtime, _, _ = mcmc.chain_rwMetropolis(np.array([4, 1]), h_metropolis,
        ban_U, num_samples, skip_n_samples)
info_str = "INFOSIMU: chain_rwMetropolis, h = " + str(h_metropolis) + \
            " runtime: " + runtime + " n_samples = " + str(num_samples) + '\n'

filename = "banana_stored_samples.smp"

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
