import numpy as np
import matplotlib.pyplot as plt
import mcmc
import nn_potential
import random
import sys
import time, datetime
    
# Potential for the neural network training
def auxiliary_tot_cost(params_together):
    # here x, y are understood as global variables defined above
    b2, b3, b4, W2, W3, W4 = nn_potential.split_params(params_together)
    return nn_potential.total_cost (nn_potential.x, 
            nn_potential.y, b2, b3, b4, W2, W3, W4, 10)

# The gradient of the potential above
def auxiliary_gradient(params_together):
    b2, b3, b4, W2, W3, W4 = nn_potential.split_params(params_together)
    return nn_potential.grad_cost (nn_potential.x,
            nn_potential.y, b2, b3, b4, W2, W3, W4)


STUDY_SINGLE_CHAIN = True #False
STUDY_CONVERGENCE = True#True

if STUDY_SINGLE_CHAIN:
    # Open the file containing the list of samples
    filename = "nn_chain.smp"
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


    # Plot the non-clustered complete energy distribution
    for i in range(23):
        plt.subplot(5, 5, i+1)
        # If 0 or 1, we are with parameter b2
        if (i < 2):
            plt.hist(X[:,i], 50, density=True, color='green')
            plt.title('b2[' + str(i+1) + ']')
        elif (i < 5):
            plt.hist(X[:,i], 50, density=True, color='darkorange')
            plt.title('b3[' + str(i-1) + ']')
        elif (i < 7):
            plt.hist(X[:,i], 50, density=True, color='magenta')
            plt.title('b4[' + str(i-4) + ']')
        elif (i < 11):
            if (i == 7):
                coeffi, coeffj = 1, 1
            elif (i == 8):
                coeffi, coeffj, = 1, 2
            elif (i == 9):
                coeffi, coeffj = 2, 1
            elif (i == 10):
                coeffi, coeffj = 2, 2
            plt.hist(X[:,i], 50, density=True, color='steelblue')
            plt.title('W2[' + str(coeffi) + ',' + str(coeffj) + ']')
        elif (i < 17):
            if (i == 11):
                coeffi, coeffj = 1, 1
            elif i == 12:
                coeffi, coeffj = 1, 2
            elif i == 13:
                coeffi, coeffj = 2, 1
            elif i == 14:
                coeffi, coeffj = 2, 2
            elif i == 15:
                coeffi, coeffj = 3, 1
            elif i == 16:
                coeffi, coeffj = 3, 2
            plt.hist(X[:,i], 50, density=True, color='grey')
            plt.title('W3[' + str(coeffi) + ',' + str(coeffj)+ ']')
        elif (i < 23):
            if (i == 17):
                coeffi, coeffj = 1, 1
            elif i == 18:
                coeffi, coeffj = 1, 2
            elif i == 19:
                coeffi, coeffj = 1, 3
            elif i == 20:
                coeffi, coeffj = 2, 1
            elif i == 21:
                coeffi, coeffj = 2, 2
            elif i == 22:
                coeffi, coeffj = 2, 3
            plt.hist(X[:,i], 50, density=True, color='darkblue')
            plt.title('W4[' + str(coeffi) + ',' + str(coeffj)+ ']')
        plt.grid(True)
    plt.suptitle("1d projections of the appoximated 23d energy landscape;" + \
        " x: values, y: probabilities\nPeaks correspond to lower total cost")
    plt.show()


    # Now, let's start with clustering
    # Sometimes we want to skip the elbow search, debug purposes
    elbow = True 
    if (elbow):
        # Find the optimal number of clusters
        mcmc.elbow_search(X, 3, 70)

    ncent = int(input("Enter the number of centroids: "))
    # Store the clusters, i.e. the points on which the distrobution concentrates
    centroids, freq = mcmc.detailed_clustering (X, ncent, auxiliary_tot_cost)


    # Plot the clustered empirical distribution
    for i in range(23):
        bb = 50# Bins
        plt.subplot(5, 5, i+1)
        # If 0 or 1, we are with parameter b2
        if (i < 2):
            plt.hist(centroids[:,i], bb, density=True, color='green')
            plt.title('b2[' + str(i+1) + ']')
        elif (i < 5):
            plt.hist(centroids[:,i], bb, density=True, color='darkorange')
            plt.title('b3[' + str(i-1) + ']')
        elif (i < 7):
            plt.hist(centroids[:,i], bb, density=True, color='magenta')
            plt.title('b4[' + str(i-4) + ']')
        elif (i < 11):
            if (i == 7):
                coeffi, coeffj = 1, 1
            elif (i == 8):
                coeffi, coeffj, = 1, 2
            elif (i == 9):
                coeffi, coeffj = 2, 1
            elif (i == 10):
                coeffi, coeffj = 2, 2
            plt.hist(centroids[:,i], bb, density=True, color='steelblue')
            plt.title('W2[' + str(coeffi) + ',' + str(coeffj) + ']')
        elif (i < 17):
            if (i == 11):
                coeffi, coeffj = 1, 1
            elif i == 12:
                coeffi, coeffj = 1, 2
            elif i == 13:
                coeffi, coeffj = 2, 1
            elif i == 14:
                coeffi, coeffj = 2, 2
            elif i == 15:
                coeffi, coeffj = 3, 1
            elif i == 16:
                coeffi, coeffj = 3, 2
            plt.hist(centroids[:,i], bb, density=True, color='grey')
            plt.title('W3[' + str(coeffi) + ',' + str(coeffj)+ ']')
        elif (i < 23):
            if (i == 17):
                coeffi, coeffj = 1, 1
            elif i == 18:
                coeffi, coeffj = 1, 2
            elif i == 19:
                coeffi, coeffj = 1, 3
            elif i == 20:
                coeffi, coeffj = 2, 1
            elif i == 21:
                coeffi, coeffj = 2, 2
            elif i == 22:
                coeffi, coeffj = 2, 3
            plt.hist(centroids[:,i], bb, density=True, color='darkblue')
            plt.title('W4[' + str(coeffi) + ',' + str(coeffj)+ ']')
        plt.grid(True)
    plt.suptitle("1d projections of the appoximated 23d energy landscape;" + \
        " x: values, y: probabilities\nPeaks correspond to lower total cost")
    plt.show()


    # Perform gradient descent on each centroid to identify the modes
    print("\nSearching for the modes... ")
    modes_list = []
    for i in range(ncent):
        print("Gradient descent on centroid number", i)
        b2, b3, b4, W2, W3, W4 = nn_potential.split_params(centroids[i])
        if(nn_potential.performGradientDescent(nn_potential.x, nn_potential.y, 
            b2, b3, b4, W2, W3, W4, eta=0.5)[0]):
            print("MODE FOUND: centroid number ", i)
            modes_list.append(i)
    
    # Now, plot all the energies for the modes found
    print("A total of " + str(len(modes_list)) + " modes have been found")
    d = len(modes_list)
    for i in range(d):
        plt.subplot(7, 10, i+1)
        a,b,c,d,e,f = nn_potential.split_params(centroids[modes_list[i]])
        nn_potential.nn_contour_plot(nn_potential.x, nn_potential.y, 
                a, b, c, d,e,f, i)
    plt.suptitle("Classification results for every mode that we found")
    plt.show()


if STUDY_CONVERGENCE:
    print("Reading the results about CONVERGENCE")
    # Open the file containing the list of samples
    filename = "nn_convergence.smp"
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

    # Computing the confidence interval for each marginal
    for i in range(m):
        # Compute the 95% confidence interval
        mean, sigma = mcmc.mean_variance1d(X[:,i])
        print("Merginal number #", i)
        print("Mean: ", mean, "sigma: ", sigma)
        print("95% Confidence Interval: [",
                        mean-2.*sigma, " ", mean+2.*sigma, "]")

    # Plot the expectations' distribution
    for i in range(m):
        plt.subplot(m, 1, i+1)
        plt.hist(X[:,i], 50, density=True)
    plt.suptitle("Convergence analysis. Gaussian = WIN")
    plt.show()
