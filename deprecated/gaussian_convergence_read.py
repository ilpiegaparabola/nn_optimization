# READ the convergence test for the GAUSSIAN sampler
import numpy as np
import matplotlib.pyplot as plt
import mcmc
import sys

# Potential for a 1dim Gaussian
def U(x):
    return x**2 / 2

def gradU(x):
    return x

# Sample candidated from HMC to minimize the potential
#X = mcmc.sample_uHMC(np.array([4]), 1, 0.1, 0.5, gradU, 10000)


# Open the file containing the list of samples
filename = "gaussian_convergence_samples.smp"
if (len(sys.argv) == 2):
    filename = str(sys.argv[1])
print("Loading:", filename)
samples_file = open(filename, "r")

# Its first line contains information about the simulation
info_str = samples_file.readline()
print(info_str[:-1])

# Collect all the samples into X
X = []
for x in samples_file:
    X.append(np.array(x[0:-1].split(' ')).astype("float64"))
X = np.asanyarray(X)
samples_file.close()
print("Read", len(X), "samples of dimension", len(X[0]))

# Plot the sampling distribution
d = len(X[0])
for i in range(d):
    plt.subplot(d, 1, i+1)
    plt.hist(X[:,i], 50, density=True)
plt.suptitle("Complete distribution")
plt.show()

# Find the optimal number of clustering
mcmc.elbow_search(X, 1, 10)
ncent = int(input("Enter the number of centroids: "))
# Store the clusters, which will be candidate modes
centroids, freq = mcmc.detailed_clustering (X, ncent, U)
# Perform gradient descent on each centroid to identify the modes
print("\nSearch for the modes: ")
for i in range(ncent):
    print("Gradient descent on candidate mode number", i)
    if(mcmc.simple_descent(centroids[i], U, gradU)):
        print("MODE FOUND: centroid number ", i)

for i in range(d):
    plt.subplot(d, 1, i+1)
    plt.scatter(centroids[:,i], freq, marker="*")
plt.suptitle("Clustered distribution")
plt.show()
