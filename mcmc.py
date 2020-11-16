import numpy as np
import random
from numpy.random import default_rng
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from numpy import log, exp, sqrt
from numpy.linalg import norm
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


#### PART 1: single steps for various chains ####

# TO RENAME AND CORRECT
# Single step (x_n, v_n) -> (x_n+1, v_n+1) for the uHMC MCMC
def uHMC (x, v, T, h, gamma, gradient):
    d = len(x)
    x, v = verlet(x, v, T, h, gradient)
    z = default_rng().multivariate_normal(np.zeros(d),np.identity(d))
    v = np.sqrt(1. - gamma * gamma) * v + gamma * z
    return x, v


# TO RENAME
# Single step x_n -> x_n+1 for the Random Walk Metropolis
def rwMetropolis(x, h, potential, L, verbose = True, local_sampler = None):
    # If None, the local_sampler corresponds to the global numpy sampler,
    # otherwise is one given from the complete chain, so that each chain
    # had a local random number generator, allowing parallelization
    d = len(x)
    print("parameter dimension: ", len(x))
    print("Starting potential: ", potential(x)[0])
    x1_complete = potential(x)[1]
    if (local_sampler == None):
        local_sampler = np.random
    y = x + sqrt(h) * \
        local_sampler.multivariate_normal(np.zeros(d), np.identity(d))
    print("Proposed potential", potential(y)[0])
    x2_complete = potential(y)[1]
    print("Norm between the old and the proposed point: ")
    print(np.linalg.norm(x1_complete - x2_complete))
    # Check that the proposed point falls into the domain
    attempts = 1
    while(not checkDomain(y, L)):
            y = x + sqrt(h) * \
                local_sampler.multivariate_normal(np.zeros(d), np.identity(d))
            attempts += 1
            if (attempts % 1000 == 0):
                print("more than", attempts, "to stay in domain")
#                input("Failed?")

    if (verbose and (attempts > 20)):
        print("Warning: more than 20 attempts to stay in domain")
    log_alpha = min(potential(x)[0] - potential(y)[0], 0)
    if log(local_sampler.uniform()) < log_alpha:
        return y, 1
    else:
        return x, 0

# TO RENAME
# Single step x_n -> x_n+1 for the Random Walk Metropolis




def rwBatchStep(x, h, potential, L, batch_size,
                                        verbose = True, local_sampler = None):
    
    batch = random.sample(range(len(x)), batch_size)
    # Update with the local sampler
    # Three auxhiliary functions for this batch case
    def from_complete_to_partial(x_complete, list_of_indeces):
        # build x_partial as a batch_size dimensional array
        # choosing the incedes on x_complete that are in batch_indeces
        length = len(list_of_indeces)
        partial_x = np.zeros(length)
        for i in range(length):
            partial_x[i] = x_complete[list_of_indeces[i]]
   #     if(verbose):
   #         print("Partial parameter: ", partial_x)
        return partial_x

    def from_partial_to_complete(x_partial, x_complete, list_of_indeces):
        # replace the inceces in x_complete, corresponding to batch_inceces
        # to the respective values in x_partial
        # temporary copy to avoid overwrite the argument
        x_copy = np.copy(x_complete)
        for i in range(len(list_of_indeces)):
            x_copy[list_of_indeces[i]] = x_partial[i]
        return x_copy

    x_small = from_complete_to_partial(x, batch)
    # If None, the local_sampler corresponds to the global numpy sampler,
    # otherwise is one given from the complete chain, so that each chain
    # had a local random number generator, allowing parallelization
    d = len(x_small)
    if(verbose):
   #     print("parameter dimension: ", len(x_small))
        print("Starting potential: ", potential(x))
    if (local_sampler == None):
        local_sampler = np.random
    y_small = x_small + sqrt(h) * \
        local_sampler.multivariate_normal(np.zeros(d), np.identity(d))

    # Check that the proposed point falls into the domain
    attempts = 1
    while(not checkDomain(y_small, L)):
            y_small = x_small + sqrt(h) * \
                local_sampler.multivariate_normal(np.zeros(d), np.identity(d))
            attempts += 1
            if (attempts % 1000 == 0):
                print("more than", attempts, "to stay in domain")
#                input("Failed?")

    if (verbose and (attempts > 20)):
        print("Warning: more than 20 attempts to stay in domain")

    if(verbose):
        print("Norm between the old and the proposed point: ")
        print(np.linalg.norm(x_small - y_small))
    y = from_partial_to_complete(y_small, x, batch)
    print("Proposed potential", potential(y))
    log_alpha = min(potential(x) - potential(y), 0)
    if log(local_sampler.uniform()) < log_alpha:
        return y, 1
    else:
        return x, 0



# Single step x_n -> x_n+1 for the Unadjusted Langevin Algorithm
def ulaStep(x, h, U, gradU, L, verbose = True, loc_sampler = None):
    # If None, the local_sampler corresponds to the global numpy sampler,
    # otherwise is one given from the complete chain, so that each chain
    # had a local random number generator, allowing parallelization
    d = len(x)
    m = np.zeros(d)
    cov = np.identity(d)
    if (loc_sampler == None):
        loc_sampler = np.random
    y = x - h*gradU(x) + np.sqrt(2.*h) * loc_sampler.multivariate_normal(m, cov)
    # Check that the proposed point falls into the domain
    attempts = 1
    while(not checkDomain(y, L)):
            y = x - h*gradU(x) + np.sqrt(2.*h) * \
                                        loc_sampler.multivariate_normal(m, cov)
            attempts += 1
            if (attempts % 1000 == 0):
                print("More than", attempts, "to stay in domain")
                input("Failed?")

    if (verbose and (attempts > 20)):
        print("Warning: more than 20 attempts to stay in domain")
    return y


def malaStep(x, h, U, gradU, L, verbose = True, loc_sampler = None):
    # If None, the local_sampler corresponds to the global numpy sampler,
    # otherwise is one given from the complete chain, so that each chain
    # had a local random number generator, allowing parallelization
    d = len(x)
    m = np.zeros(d)
    cov = np.identity(d)
    if (loc_sampler == None):
        loc_sampler = np.random
    y = x - h*gradU(x) + np.sqrt(2.*h) * loc_sampler.multivariate_normal(m, cov)
    # Check that the proposed point falls into the domain
    attempts = 1
    while(not checkDomain(y, L)):
            y = x - h*gradU(x) + np.sqrt(2.*h) * \
                                        loc_sampler.multivariate_normal(m, cov)
            attempts += 1
            if (attempts % 1000 == 0):
                print("More than", attempts, "to stay in domain")
                input("Failed?")
       
    if (verbose and (attempts > 20)):
        print("Warning: more than 20 attempts to stay in domain")
    # Now correct with a Matropolis step
    Gxy = U(y) - U(x) - np.dot(y-x, gradU(x) + gradU(y)) / 2. + \
            (h/4.) * (norm(gradU(y))**2. - norm(gradU(x))**2)
    #print("Gxy :", Gxy)
    #input("..")
    log_alpha = max( - Gxy, 0)
    if log(loc_sampler.uniform()) < log_alpha:
        return y, 1
    else:
        return x, 0



### PART 2: Complete Monte Carlo Chains #####

# Complete ULA chain
def ulaChain(start_x, h, U, gradU, n_samples,
        skip_rate=5, L=10, verbose = True, seed = None):
    # Local random sampler, to allow parallelization
    chain_sampler = np.random.RandomState(seed)
    # Burning-time rate. 5 = 20%
    bt_rate = 5
    # n_samples is the length of the chain with no burning time
    total_samples = bt_rate * n_samples / (bt_rate - 1)
    # use the bt_rate to obtain the number of discarded samples
    bt = int (total_samples / bt_rate)
    # Run a single chain step and compute the expected running time
    start_time = time.time()
    xnew  = ulaStep(start_x, h, U, gradU, L,verbose, chain_sampler)
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
        xnew = ulaStep(xnew, h, U, gradU, L, verbose, chain_sampler)

    # Produce the first valid sample, and start counting acceptance rate
    if (verbose):
        print("Markov chain started!")
    x_samples = []
    xnew = ulaStep(xnew, h, U, gradU, L, verbose, chain_sampler)
    x_samples.append(xnew)

    # Append all the remaining valid samples, one every skip_rate samples
    for i in range(1, n_samples):
        for l in range(skip_rate):
            xnew = ulaStep(xnew,h, U, gradU, L, verbose, chain_sampler)
        x_samples.append(xnew)
#        void = input("DEBUG: press enter for the next sample")
        if (verbose and (i % 2000 == 0)):
            print("Sample #", i)
#            print(x_samples[i])
    if (verbose):
        print("--- end of the chain ---\nBurning samples: ", bt)
        print("Skip rate: ", skip_rate, "\nEffective samples: ", n_samples)
        print("Total samples: ", bt+n_samples*skip_rate,
                " = burning_samples + ", "skip_rate * Effective samples")

    runtime = str(datetime.timedelta(seconds = int(time.time()-start_time)))+ \
       " skip_rate = " + str(skip_rate) + "Domain: " + str(L)

    if (verbose):
        print("Actual duration = " + runtime)
    x_samples = np.asanyarray(x_samples) 
    expect = sum([x for x in x_samples]) / len(x_samples)
    print("Chain expectation: ", expect)
    return x_samples, runtime, expect

 
# Complete MALA chain
def malaChain(start_x, h, U, gradU, n_samples,
        skip_rate=5, L=10, verbose = True, seed = None):
    # Local random sampler to allow parallelization
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
    xnew, is_accepted  = malaStep(start_x, h, U, gradU, L, 
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
        xnew, is_accepted  = malaStep(xnew, h, U, gradU, L, 
                                                verbose, chain_sampler)

    # Produce the first valid sample, and start counting acceptance rate
    if (verbose):
        print("Markov chain started!")
    x_samples = []
    xnew, is_accepted = malaStep(xnew, h, U, gradU, L,
                                                verbose, chain_sampler)
    accept_rate += is_accepted
    x_samples.append(xnew)

    # Append all the remaining valid samples, one every skip_rate samples
    for i in range(1, n_samples):
        for l in range(skip_rate):
#        xnew, is_accepted = rwMetropolis(x_samples[i-1], h, potential)
            xnew, is_accepted = malaStep(xnew,h, U, gradU, L,
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
    return x_samples, runtime, accept_rate, expect


# TO REWRITE
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

# Complete Random Walk Metropolis chain
def chain_rwBatchMetropolis(start_x, h, potential, n_samples, batch_size,
        skip_rate=5, L=10, verbose = True, seed = None):
    chain_sampler = np.random.RandomState(seed)
    # Burning-time rate. 5 = 20%
    bt_rate = 7
    # n_samples is the length of the chain with no burning time
    # We compute total_samples, the total lentgh included the burning time
    total_samples = bt_rate * n_samples / (bt_rate - 1)
    # use the bt_rate to obtain the number of discarded samples
    bt = int (total_samples / bt_rate)

    # Run a single chain step and compute the expected running time
    start_time = time.time()
    accept_rate, is_accepted = 0, 0
    xnew, is_accepted  = rwBatchStep(start_x, h, potential, L, batch_size, 
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
        xnew, is_accepted  = rwBatchStep(xnew, h, potential, L, batch_size,
                                                verbose, chain_sampler)

    # Produce the first valid sample, and start counting acceptance rate
    if (verbose):
        print("Markov chain started!")
    x_samples = []
    xnew, is_accepted = rwBatchStep(xnew, h, potential, L, batch_size,
                                                verbose, chain_sampler)
    accept_rate += is_accepted
    x_samples.append(xnew)

    # Append all the remaining valid samples, one every skip_rate samples
    for i in range(1, n_samples):
        for l in range(skip_rate):
#        xnew, is_accepted = rwMetropolis(x_samples[i-1], h, potential)
            xnew, is_accepted = rwBatchStep(xnew,h,potential, L, batch_size,
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
    if verbose:
        print("Chain expectation: ", expect)
    return x_samples, runtime, accept_rate, expect


# Complete Random Walk Metropolis chain
def chain_rwMetropolis(start_x, h, potential, n_samples,
        skip_rate=5, L=10, verbose = True, seed = None):
    chain_sampler = np.random.RandomState(seed)
    # Burning-time rate. 5 = 20%
    bt_rate = 7
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
    if verbose:
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


# Let's try with a multi-chain approach as described in my notes. vbs = verbose
def multichainRW(dimx, L, h, pot, n_samples, n_chains, thinning, vbs = False):
    if vbs:
        print("---- Multichain approach: RW Metropolis ----")
    chains = []
    arates = []
    def add_chain(metropolisRW_result):
        chains.append(metropolisRW_result[0])
        arates.append(metropolisRW_result[2])
    
    # Prepare #n_chains random starting points
    st_points = []
    for i in range(n_chains):
        st_points.append(np.random.uniform(-L, L, dimx))

    # Run a single chain just to give a time estimation
    start_time = time.time()
    add_chain(chain_rwMetropolis(st_points[0], h, pot, n_samples, 
                                                    thinning, L, False, None))
    if vbs:
        linear_run_time = int((time.time() - start_time) * n_chains)
        print("Approximated MAX running time: " + \
                            str(datetime.timedelta(seconds = linear_run_time)))
        print("Approx. MIN running time: " + \
           str(datetime.timedelta(seconds = linear_run_time / mp.cpu_count())))

    # Run multiple chains in parallel, each stored in chains[]
    pool = mp.Pool(mp.cpu_count())
    for j in range(1, n_chains):
        pool.apply_async(chain_rwMetropolis,
                     args = (st_points[j],h,pot,n_samples,thinning,L,False,j),
                                                          callback = add_chain)
    pool.close()
    pool.join()

    # Now construct and a return a single chain of n_samples from the parall.
    X = []
    for i in range(n_samples):
        # Add to X a random sample from a random chain, counting from the end
        # (in order to avoid the burning time samples)
        nth = np.random.random_integers(1, n_samples - 1)
        X.append(chains[np.random.random_integers(0, n_chains -1)][-nth])
    mean_acceptance = sum(arates) / len(arates)
    print("Averge rate: ", mean_acceptance)
    expect = sum([x for x in X]) / n_samples
    print("Multichain expectation: ", expect)
    return X, mean_acceptance, expect


################################################################
#################### PART 3: CONVERGENCE #################
#########################################################

# Convergence for the random walk multichain metropolis
def multichainRWconvergence(dimx, L, h, pot, n_samples, n_chains, thinning, 
        n_conv):
    print("--- CONVERGENCE of multichain RW method ---")
    print("(each chain the combination of", n_chains, "chains)")
    # Just run n_conv instances of multichainRW and take their expectations
    expectations = []
    # Run a single chain just to give a time estimation
    start_time = time.time()
    expectations.append(multichainRW(dimx, L, h, pot, 
                                            n_samples, n_chains, thinning)[2])
    linear_run_time = int((time.time() - start_time) * n_conv)
    print("Approximated running time: " + \
            str(datetime.timedelta(seconds = linear_run_time)))
    for i in range(1, n_conv):
        print("Chain ", i+1, "of", n_conv)
        expectations.append(multichainRW(dimx, L, h, pot, 
                                            n_samples, n_chains, thinning)[2])
    return expectations


def convergenceMetropolis(start_x, h, pot, n_samples, 
        skip_rate, L, conv_samples, parallel = True):
    print(" ---- Samples for studying CONVERGENCE ---- ")
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


# Checking the convergence for the ULA implementation
def ulaConvergence(start_x, h, U, gradU, n_samples, 
        skip_rate, L, conv_samples, parallel = True):
    print(" ---- Samples for studying ULA CONVERGENCE ---- ")
    print("Expectations of", conv_samples, "Markov Chains.")
    print("Running the first Markov chain...")

    # List of all the computed expectations and a function to append to them
    # the results of a metropolis run. mcmc[2] is the expectation
    expectations = []
    def add_expect(mcmc_result):
        expectations.append(mcmc_result[2])

    # Run conv_samples Metropolis instance, storing all the expectation vals
    start_time = time.time()
    add_expect(ulaChain(start_x, h, U, gradU, n_samples, 
        skip_rate, L, False, None))
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
            pool.apply_async(ulaChain,
                    args=(start_x, h, U, gradU, 
                        n_samples, skip_rate, L, False, j),
                    callback=add_expect)
        pool.close()
        pool.join()
    else:
        print("WARNING: NOT PARALLELIZED")
        for i in range(1, conv_samples):
            print("***CONV*** Expectation sample # ", i)
            add_expect(ulaChain(start_x, h, U, gradU, pot, n_samples, 
                                    skip_rate, L, False, None))
    print("Actual running time: ", 
            str(datetime.timedelta(seconds=int(time.time() - start_time))))
    # Return the list of all the expectations, and the avergace accp rate
    return expectations


# Convergence in the case of MALA 
def malaConvergence(start_x, h, U, gradU, n_samples, 
        skip_rate, L, conv_samples, parallel = True):
    print(" ---- Samples for studying MALA CONVERGENCE ---- ")
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
    add_expect(malaChain(start_x, h, U, gradU, n_samples, 
        skip_rate, L, False, None))
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
            pool.apply_async(malaChain,
                    args=(start_x, h, U, gradU, 
                        n_samples, skip_rate, L, False, j),
                    callback=add_expect)
        pool.close()
        pool.join()
    else:
        print("WARNING: NOT PARALLELIZED")
        for i in range(1, conv_samples):
            print("***CONV*** Expectation sample # ", i)
            add_expect(malaChain(start_x, h, U, gradU, n_samples, 
                                    skip_rate, L, False, None))
    print("Actual running time: ", 
            str(datetime.timedelta(seconds=int(time.time() - start_time))))

    average_accept = sum(accp_rates) / len(accp_rates)
    print ("Average acceptance rate: ", average_accept, "%")
    # Return the list of all the expectations, and the avergace accp rate
    return expectations, average_accept


########################################################
######### PART 4 : empirical data analysis #############
########################################################

# Given a list of 1-dimensional samples, return the confidence interval
def mean_variance1d(samples1d):
    mean = np.mean(samples1d)
    sigma = 0.
    n = len(samples1d)
    for i in range(n):
        sigma += (samples1d[i] - mean) ** 2.
    sigma = np.sqrt(sigma / (n - 1))
    return mean, sigma



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

