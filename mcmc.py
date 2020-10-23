import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from numpy import log, exp, sqrt
import time, datetime
import multiprocessing as mp

# Check that a point is into an hypercube of lato -L, L
def checkDomain (x, L=10):
    for p in x:
        if (p < -L or p > L):
            return False
    return True


# Verlet integrator from time 0 until T
def verlet(x, v, T, h, gradient):
    # From time 0 to T, define so k = T / h
    k = int(T / h)
    for i in range(k):
        x_old = np.copy(x)
        x = x_old + h * v - h * h * gradient(x_old) / 2.
        v = v - h / 2. * gradient(x_old) - h / 2. * gradient(x)
    return x, v


# Single step (x_n, v_n) -> (x_n+1, v_n+1) for the uHMC MCMC
def uHMC (x, v, T, h, gamma, gradient):
    d = len(x)
    x, v = verlet(x, v, T, h, gradient)
    z = default_rng().multivariate_normal(np.zeros(d),np.identity(d))
    v = np.sqrt(1. - gamma * gamma) * v + gamma * z
    return x, v

   
#Complete uHMC sampler, i.e. repeating uHMC multiple times
def chain_uHMC(start_x, T, h, gamma, gradient, n_samples):
    # Burning-time rate. 5 = 20%
    bt_rate = 5
    # n_samples is the actual used length of the chain, compute total_samples
    # to obtain the total lentgh included the burning time
    total_samples = bt_rate * n_samples / (bt_rate - 1)
    # take the bt_rate to obtain the number of discarded samples
    bt = int (total_samples / bt_rate)

    # Generate the velocity's staring value
    d = len(start_x)
    start_v = default_rng().multivariate_normal(np.zeros(d),np.identity(d))

    start_time = time.time()
    # Run a single chain step
    current_sample = uHMC(start_x, start_v, T, h, gamma, gradient)
    time_one_sample = time.time() - start_time
    total_running_time = int(total_samples * time_one_sample) 
    print("Expected total running time: ", 
            str(datetime.timedelta(seconds = total_running_time)))
    print("Expected burning time...", 
            str(datetime.timedelta(seconds = int(time_one_sample * (bt-1)))))
    for i in range(bt - 1):
        current_samples = uHMC(current_sample[0], current_sample[1], T, 
                                                h, gamma, gradient)
    print("Starting the chain!")
    x_samples = []
    v_samples = []
    xnew, vnew = uHMC(current_sample[0], current_sample[1], T, h, gamma,                                                                  gradient)
    x_samples.append(xnew)
    v_samples.append(vnew)
    # After the burning time, consider all the samples
    for i in range(1, n_samples):
        xnew, vnew = uHMC(x_samples[i-1],v_samples[i-1], T, h, gamma, gradient)
        x_samples.append(xnew)
        v_samples.append(vnew)
        if (i % 50 == 0):
            print("Sample #", i, end=' ')
            print(x_samples[i])
    print("--- end of the chain ---")
    print("Total samples: ", total_samples)
    print("Burning time: ", bt)
    print("Effective samples: ", n_samples)
    runtime = \
        str(datetime.timedelta(seconds = int(time.time()-start_time)))
    print("Actual duration = " + runtime)
    return np.asanyarray(x_samples), runtime

# This function is deprecated, but leave the name in order not to change
# every single file refering to it. But I will do.
def sample_uHMC(start_x, T, h, gamma, gradient, n_samples):
    return chain_uHMC(start_x, T, h, gamma, gradient, n_samples)


# Single step x_n -> x_n+1 for the Random Walk Metropolis
def rwMetropolis(x, h, potential, L=10, verbose = True, local_sampler = None):
    # If None, the local_sampler corresponds to the global numpy sampler,
    # otherwise is one given from the complete chain, so that each chain
    # had a local random number generator, allowing parallelization
    if (local_sampler == None):
        local_sampler = np.random
    y = x + sqrt(h) * \
        local_sampler.multivariate_normal(np.zeros(len(x)), np.identity(len(x)))
        #default_rng().multivariate_normal(np.zeros(len(x)),np.identity(len(x)))
# Check that the proposed point falls into the domain
    attempts = 1
    while(not checkDomain(y, L)):
            y = x + sqrt(h) * \
        local_sampler.multivariate_normal(np.zeros(len(x)), np.identity(len(x)))
     #default_rng().multivariate_normal(np.zeros(len(x)), 
            #                                            np.identity(len(x)))
            attempts += 1
        
    if (verbose and (attempts > 20)):
        print("Warning: more than 20 attempts to stay in domain")

    log_alpha = min(potential(x) - potential(y), 0)
    if log(local_sampler.uniform()) < log_alpha:
        return y, 1
    else:
        return x, 0


# Alternative for debug
def DEBUG_rwMetropolis(x, h, potential):
    print("Starting energy: ", potential(x))
    y = x + sqrt(h) * \
        default_rng().multivariate_normal(np.zeros(len(x)), np.identity(len(x)))
    alpha = min(1, exp(potential(x) - potential(y)))
    print("Final energy: ", potential(y))
    p = np.random.uniform()
    print("alpha = ", alpha, "p = ", p)
    if (p  < alpha):
        print("Accepted")
        return y, 1
    else:
        return x, 0


# Complete Random Walk Metropolis chain
def chain_rwMetropolis(start_x, h, potential, n_samples,
        skip_rate=5, L=10, verbose = True, seed = None):
    chain_sampler = np.random.RandomState(seed)
    # Burning-time rate. 5 = 20%
    bt_rate = 5
    # n_samples is the length of the chain with no burning time
    # We compute total_samples, the total lentgh included the burning time
    total_samples = bt_rate * n_samples / (bt_rate - 1)
    # use the bt_rate to obtain the number of discarded samples
    bt = int (total_samples / bt_rate)

    # Run a single chain step and compute the expected running time
    start_time = time.time()
    accept_rate, is_accepted = 0, 0
    xnew, is_accepted  = rwMetropolis(start_x, h, potential, L, 
                                        verbose, chain_sampler)
    time_one_sample = time.time() - start_time

    time_burning = time_one_sample * (bt - 1) 
    time_total = time_burning + n_samples * time_one_sample * skip_rate
    if (verbose):
        print("Approximated total running time: ", 
            str(datetime.timedelta(seconds = int(time_total))))
        print("Approximated burning time...", 
            str(datetime.timedelta(seconds = int(time_burning))))
#    input("Press ENTER to proceed.")
        print("Burning time started...")
    for i in range(bt - 1):
        xnew, is_accepted  = rwMetropolis(xnew, h, potential, L, 
                                                verbose, chain_sampler)

    # Produce the first valid sample, and start counting acceptance rate
    if (verbose):
        print("Markov chain started!")
    x_samples = []
    xnew, is_accepted = rwMetropolis(xnew, h, potential, L,
                                                verbose, chain_sampler)
    accept_rate += is_accepted
    x_samples.append(xnew)

    # Append all the remaining valid samples, one every skip_rate samples
    for i in range(1, n_samples):
        for l in range(skip_rate):
#        xnew, is_accepted = rwMetropolis(x_samples[i-1], h, potential)
            xnew, is_accepted = rwMetropolis(xnew,h,potential, L,
                                                verbose, chain_sampler)
            accept_rate += is_accepted
        x_samples.append(xnew)
#        void = input("DEBUG: press enter for the next sample")
        if (verbose and (i % 2000 == 0)):
            print("Sample #", i)
#            print(x_samples[i])

    accept_rate = int(accept_rate * 100. / (n_samples * skip_rate))
    if (verbose):
        print("--- end of the chain ---\nBurning samples: ", bt)
        print("Skip rate: ", skip_rate, "\nEffective samples: ", n_samples)
        print("Total samples: ", bt+n_samples*skip_rate,
                " = burning_samples + ", "skip_rate * Effective samples")

    runtime = str(datetime.timedelta(seconds = int(time.time()-start_time)))+ \
       " accept_rate: " + str(accept_rate) + "%, skip_rate = " + str(skip_rate)\
       + "Domain: " + str(L)

    if (verbose):
        print("Actual duration = " + runtime)
    x_samples = np.asanyarray(x_samples) 
    expect = sum([x for x in x_samples]) / len(x_samples)
    print("Chain expectation: ", expect)
    ### TEMPORARELY HERE, for the Neural Network case
#    print("EXPECTATION: ")
##    b2 = expect[0:2]
#    b3 = expect[2:5]
#    b4 = expect[5:7]
#    W2 = expect[7:11].reshape(2, 2)
#    W3 = expect[11:17].reshape(3, 2)
#    W4 = expect[17:23].reshape(2, 3)
#    print("b2 = ", b2)
#    print("W2 = ", W2)
#    print("b3 = ", b3)
#    print("b4 = ", b4)
#    print("W3 = ", W3)
#    print("W4 = ", W4)
    return x_samples, runtime, accept_rate, expect


def convergenceMetropolis(start_x, h, pot, n_samples, 
        skip_rate, L, conv_samples, parallel = True):
    print(" ---- CONVERGENCE ANALYSIS ---- ")
    print("Expectations of", conv_samples, "Markov Chains.")
    print("Running the first Markov chain...")

    # List of all the computed expectations and a function to append to them
    # the results of a metropolis run. mcmc[2] is the expectation
    expectations = []
    accp_rates = []
    def add_expect(mcmc_result):
        expectations.append(mcmc_result[3])
        accp_rates.append(mcmc_result[2])

    # Run conv_samples Metropolis instance, storing all the expectation vals
    start_time = time.time()
    add_expect(chain_rwMetropolis(start_x, h, pot, n_samples, 
        skip_rate, L, False, None))

    #expectations.append(chain_rwMetropolis(start_x, h, pot, n_samples, 
    #    skip_rate, L, False, None)[2])
    linear_run_time = int((time.time() - start_time) * conv_samples) 
    print("Approximated MAX running time: " + \
            str(datetime.timedelta(seconds = linear_run_time)))

    # Running conv_samples mcmc chains
    if (parallel):
        print("Parallelized!")
        print("Approx. MIN running time: " + \
           str(datetime.timedelta(seconds = linear_run_time / mp.cpu_count())))
        pool = mp.Pool(mp.cpu_count())
        for j in range(1, conv_samples):
            pool.apply_async(chain_rwMetropolis,
                    args=(start_x, h, pot, n_samples, skip_rate, L, False, j),
                    callback=add_expect)
        pool.close()
        pool.join()
    else:
        print("WARNING: NOT PARALLELIZED")
        for i in range(1, conv_samples):
            print("***CONV*** Expectation sample # ", i)
            add_expect(chain_rwMetropolis(start_x, h, pot, n_samples, 
                                    skip_rate, L, False, None))
    print("Actual running time: ", 
            str(datetime.timedelta(seconds=int(time.time() - start_time))))

    average_accept = sum(accp_rates) / len(accp_rates)
    print ("Average acceptance rate: ", average_accept, "%")
    # Return the list of all the expectations, and the avergace accp rate
    return expectations, average_accept


# Given a set of samples X, find out the optimal number of centroids
def elbow_search(X, min_modes=2, max_modes=20):
    print("Performing various k-means clustering: elbow search")
    print("Going from ", min_modes, "to", max_modes-1, "centroids")
    elbow = []
    for i in range(min_modes, max_modes):
        print("Current:", i, "centroids")
        k_means = KMeans(init='k-means++', n_clusters=i, n_init=12)
        k_means.fit(X)
        elbow.append(k_means.inertia_)
    plt.plot(range(min_modes, max_modes), elbow)
    plt.title("Elbow method for researching the optimal modes")
    plt.show()


# Given a set of samples X, cluster it with n_centrods and display
# informations like where the centers are located, their distance,
# their potential values, their frequency, etc...
def detailed_clustering (X, n_centroids, potential, plot_cluster_freq=False):

    kmeans = KMeans(n_clusters = n_centroids, n_init=20, max_iter=5000).fit(X)
    # Convert kmeans.labels_ into an histogram, to display the frequencies
    # In principle, if the probability space has been well explored,
    # we expect similar frequencies for each cluster.
    kmeans_histo = np.zeros(n_centroids)
    for i in range(len(X)):
        kmeans_histo[kmeans.labels_[i]] += 1

    # Format the histogram to contain frequencies
    for i in range(len(kmeans_histo)):
        kmeans_histo[i] = kmeans_histo[i] * 100. / len(X)

    # Visualize the centroids and their evaluations
    print("----------------------------------------")
    print("Labels VS Potential value VS Frequencies")
    for i in range(n_centroids):
        print(i, potential(kmeans.cluster_centers_[i]), kmeans_histo[i], "%")
    
    print("Additional Information about the centroids:")
    print("Labels VS Centroids VS Potential value VS Frequencies")
    for i in range(n_centroids):
        print(i, kmeans.cluster_centers_[i], 
                potential(kmeans.cluster_centers_[i]), kmeans_histo[i], "%")

    #Computing my distance matrix
    d_mtx = np.identity(n_centroids)
    for i in range(n_centroids):
        for j in range(n_centroids):
            d_mtx[i][j] = np.linalg.norm(kmeans.cluster_centers_[i] - 
                    kmeans.cluster_centers_[j])
    print("Distances between centers i-j:")
    print(d_mtx)
    
    # Plot an histogram of the frequencies: the centroids with highest
    # frequencies are interpreted as modes
    if (plot_cluster_freq):
        plt.scatter(range(n_centroids), kmeans_histo)
        plt.show()
    return kmeans.cluster_centers_, kmeans_histo


def simple_descent(x, U, gradient, eta=0.5, steps=1000, mode_tol=0.05):
    x_old = np.copy(x)
    print("Starting cost: ", U(x))
    for i in range(steps):
        x = x - eta * gradient(x)
    print("Final cost: ", U(x))
#    print("New point distance: ", np.linalg.norm(x - x_old))
    cost_reduction = U(x_old) - U(x)
    print("Cost reduction (the higher, the less likely it's a mode): ", 
            cost_reduction)
    if cost_reduction < mode_tol:
        return True
    else:
        return False

