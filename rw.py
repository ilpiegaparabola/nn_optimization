# Mini-library dedicated to the Random Walk Metropolis algorithm
# more explanations and comments will follow
import numpy as np
import random
from numpy import log, exp, sqrt
from numpy.linalg import norm
import time, datetime
import multiprocessing as mp
from numpy import identity as I
from datetime import timedelta as tdelta

# Check that a point is into an hypercube of lenghts 2L
def checkDomain (x, L=10):
    for p in x:
        if (p < -L or p > L):
            return False
    return True


# A single step x_n -> x_n+1 for the Random Walk Metropolis algorithms
# Parameters:
# x             : the starting point of dimension, say, D
# h             : float, the covariance step
# pot           : function R^D -> R to minimize / Distribution to sample from
# L             : the samples are checked to be inside a cube of lenght 2L
# verbose       : enable info
# loc_sampler   : np.random for default, but it can be set in order to
#                   allow different seeds needed for parallelization
# RETURN        : the couple (x_n, 0), rejection, or (x_n+1, 1), acceptance.
def stepRW (x, h, pot, L, verbose = True, loc_sampler = np.random):
    d = len(x)
    y = x + sqrt(h) * loc_sampler.multivariate_normal(np.zeros(d), I(d))

    # Check that the proposed point falls into the domain
    attempts = 1
    while(not checkDomain(y, L)):
            y = x + sqrt(h)*loc_sampler.multivariate_normal(np.zeros(d), I(d))
            attempts += 1
            if (verbose and attempts % 1000 == 0):
                print("!!! currently, more than", attempts, "to stay in domain")
    if (verbose and (attempts > 20)):
        print("Warning: more than 20 attempts _done_ to stay in domain")

    # Determine if to accept the new point or not
    if (verbose):   
        print("Pot(x): ", int(pot(x)), "Pot.(y): ", 
                int(pot(y)), "x-y = ", norm(x-y))
    log_alpha = min(pot(x) - pot(y), 0)
    if log(loc_sampler.uniform()) < log_alpha:
        return y, 1
    else:
        return x, 0


# A Random Walk Metropolis chain, i.e. the iteration of the single step 
# described in the function stepRW
# PARAMETERS
# startx    : (narray) point where to starts. Say, it has dimension D
# h         : (float) the covariance step
# pot       : function R^D -> R to minimize / Distribution to sample from
# nsamples  : (int) number of samples we want. Related to the chain lenght
# thin      : (int) thinning size. 1 = no thinnig
# L         : (float) samples are checked to be in a cube of lenght 2L
# verbose   : (int) 0 no messages, 1 local messages, 2 messages from stepRW,too
# loc_seed  : None as default. When integer, enable a local seed for parallel.
# RETURNS   : (chain, infos, acceptance rate, expectation value)
def chainRW (startx, h, pot, nsamples, thin=1, L=5, verbose=2, loc_seed =None):
    copy_x = np.copy(startx)
    chain_sampler = np.random.RandomState(loc_seed)
    # Burning rate. 5 = 20%
    brate = 5
    # nsamples is basically the lenght of the chain without burning time
    # we need to compute the total lenght, too, to give time running estimation
    totsamples = brate * nsamples / (brate - 1.)
    # number of burned samples:
    bsamples = int (totsamples / brate)
    # Chain accept rate, and variable to detect if proposal is accepted
    acceptrate = 0
    isaccepted = 0    
    # Produce a single sample, with time estimation
    start_time = time.time()
    xnew, isaccepted = stepRW(startx, h, pot, L, verbose - 1, chain_sampler)
    acceptrate += isaccepted
    timesample = time.time() - start_time
    btime = timesample * (bsamples - 1)
    timetotal = btime + nsamples * timesample * thin
    if (verbose):
        print("Approx. _total_ time: ", str(tdelta(seconds = int(timetotal))))
        print("Approx. burning time: ", str(tdelta(seconds = int(btime))))
        print("...burning time started.")
    for i in range(bsamples - 1):
        xnew, isaccepted = stepRW(xnew, h, pot, L, verbose-1, chain_sampler)
        acceptrate += isaccepted
    if (verbose):
        print("burning time ended. Actual Markov Chain started.")
    xsamples = []
    # From now, consider the thinning rate (i.e. skip every thin-1 samples)
    for i in range(nsamples):
        for l in range(thin):
             xnew, isaccepted = stepRW(xnew,h,pot,L,verbose-1, chain_sampler)
             acceptrate += isaccepted
        xsamples.append(xnew)
        if (verbose and (i % 1000 == 0)):
            print("Sample #", i)
    # Information to return, useful to reproduce the results
    xsamples = np.asanyarray(xsamples)
    acceptrate = acceptrate * 100. / (bsamples + nsamples * thin)
    expect = sum([x for x in xsamples]) / len(xsamples)
    #expect = xsamples.mean() <- WRONG
    info = str(tdelta(seconds=int(time.time()-start_time))) + " accept_rate" +\
           str(acceptrate) + "%, thinning: " + str(thin)+ "Domain: "+ str(L)
    if (verbose):
        print("--- end of the chain---\nBurned samples: ", bsamples)
        print("Thinning rate: ", thin, "\nEffective samples: ", nsamples)
        print("Total chain lenght: ", totsamples)
        print("Chain expectations: ", expect)
        print("Acceptance rate: ", acceptrate, "%")
        print("h: ", h)
        print("L: ", L)
        print("Starting point: ", copy_x)
    return xsamples, info, acceptrate, expect


#--- DEBUG FUNCTION
### Temporarely dummy function which overwrites chainRW
#def chainRW(x, h, pot, nsamples, thin, L, values, j):
#    return [0, 1, 2]
#--------------------------------------------------------#


# Multichain RW: a way to produce various RW chains _in parallel_, and then
# taking samples from them, randomly, with the hope of creating a single
# chain with a modest correlation value.
# PARAMETERS
# dimx      : (int) dimension of every samples
# h         : (float) the covariance step
# pot       : function R^D -> R to minimize / Distribution to sample from
# nsamples  : (int) number of samples we want. Related to the chain lenght
# nchains   : (int) number of chains to produce in parallel
# thin      : (int) thinning size. 1 = no thinnig
# L         : (float) samples are checked to be in a cube of lenght 2L
# verbose   : 
# RETURNS   : (chain, infos, acceptance rate, expectation value)
def multiRW (dimx, h, pot, nsamples, nchains, thin, L, verbose = True):
    parallelchains = mp.cpu_count()
    if (verbose):
        print("--- Multichain MC ---")
        print("TOTAL number of chains: ", nchains)
        print("Chains in parallel: ", parallelchains)
        print("samples of dimension ", dimx)
#        input("press ENTER to continue")
    chains = []
    acceptrates = []
    # Function to read the results produced by a single chainRW
    def addChain (chainRW_result):
        chains.append(chainRW_result[0])
        acceptrates.append(chainRW_result[2])
    # Prepare #nchains random starting point, each is a startx for a chain
    startpoints = []
    for i in range(nchains):
        startpoints.append(np.random.uniform(-L, L, dimx))
    # Run a single chain to have a time estimation
    starttime = time.time()
    addChain(chainRW(startpoints[0], h, pot, nsamples, thin, L, False, None))
    if (verbose):
        linear_time = int((time.time() - starttime) * nchains)
        optim_time = linear_time / parallelchains
        print("Approx. MAX running time: " + str(tdelta(seconds= linear_time)))
        print("Approx. MIN running time: " + str(tdelta(seconds = optim_time)))
        print("staring with parallels chains")
    # Produce multiple chains in parallel
#    print("FLAG1")
    pool = mp.Pool(parallelchains)
#    print("FLAG2")
    for j in range(1, nchains):
        pool.apply_async(chainRW, args = (startpoints[j], h, pot, nsamples,
                                      thin, L, False, j), callback = addChain)
    pool.close()
    pool.join()
#    print("FLAG3")
#    if (verbose):
#        print("Chains = ", chains)
#        input("Press ENTER: building the single final chain.")
    # Now, construct a SINGLE chain by taking random samples from the chains
    X = []
    for i in range(nsamples):
        # Add to X a random sample from a random chain, counting from the end
        # (in order to avoid the burning time samples)
        nth = np.random.random_integers(1, nsamples - 1)
        X.append(chains[np.random.random_integers(0, nchains -1)][-nth])
    X = np.asanyarray(X)
    acceptrates = np.asanyarray(acceptrates)
    mean_acceptance = sum([x for x in acceptrates]) / len(acceptrates)
    print("Averge rate: ", mean_acceptance)
    expect = sum([x for x in X]) / nsamples
    print("Multichain expectation: ", expect)
    return X, mean_acceptance, expect



#### Run multiple instances of Monte Carlo chains and collect their
# expectations. It is a tool to check convergence: if they converge,
# they must be gaussians in every marginal. Such a checking is done 
# independently. The following functions just run the multi
