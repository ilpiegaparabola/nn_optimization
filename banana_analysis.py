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


# Open the file containing the list of samples
filename = "banana_stored_samples.smp"
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

# Find the optimal number of clustering
#mcmc.elbow_search(X, 1, 20)
ncent = int(input("Enter the number of centroids: "))
# Store the clusters, which will be candidate modes
centroids, freq = mcmc.detailed_clustering (X, ncent, ban_U)
freq = freq / 100.
# Perform gradient descent on each centroid to identify the modes
print("\nSearch for the modes: ")
for i in range(ncent):
    print("Gradient descent on candidate mode number", i)
    if(mcmc.simple_descent(centroids[i], ban_U, ban_gradU, eta=0.001)):
        print("MODE FOUND: centroid number ", i)

# Plot the sampling empirical distribution
d = len(X[0])
for i in range(d):
    plt.subplot(d, 1, i+1)
    plt.hist(X[:,i], 100, density=True, color='steelblue')
    plt.grid(True)
plt.suptitle("Complete distribution")
plt.show()

for i in range(d):
    plt.subplot(d, 1, i+1)
    plt.scatter(centroids[:,i], freq, marker="*", color='green')
    plt.grid(True)
plt.suptitle("Clustered distribution")
plt.show()
