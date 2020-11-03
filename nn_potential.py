import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import sys

### Part 1: useful functions

def activation (x):
    # Sigmoid function
    return 1 / (1 + np.exp(-x))

#derivative of the sigmoid activation function
def der_activation (x):
    return activation(x)*(1 - activation(x))

def hadamart (v1, v2):
    res = np.ones(len(v1))
    for i in range(len(v1)):
        res[i] = v1[i]*v2[i]
    return res

# Given a 2dim point x_i, composing classify(NN(x)) determines its category
# In Nigham's notation, it is y(x_i)
def classify (pt):
    if pt[0] > pt[1]:
        return np.array([1, 0])
    else:
        return np.array([0, 1])

def NN_to_scalar (x):
    if x[0] > x[1]:
        return 1
    else:
        return 0

# Our very simple NN evaluated on a single 2dim point x_pt
def NN (x_pt, b2, b3, b4, W2, W3, W4):
    z2 = np.dot(W2, x_pt) + b2
    a2 = activation(z2)
    z3 = np.dot(W3, a2) + b3
    a3 = activation(z3)
    z4 = np.dot(W4, a3) + b4
    return (activation(z4))


#### TO CHECK WHAT FOLLOWS
# The total cost function (i.e. the potential used in HMC)
def total_cost (x, y, b2, b3, b4, W2, W3, W4, N=10):
    result = np.array([np.linalg.norm(y[i] - NN(x[i], b2, b3, b4, W2, W3, W4))
                for i in range(N)])
    return np.mean(result)

# Gradient of the i-th component of the cost function, computed by
# using back propagation.
def ith_grad_cost (x_pt, y_pt, b2, b3, b4, W2, W3, W4):
    z2 = np.dot(W2, x_pt) + b2
    a2 = activation(z2)
    z3 = np.dot(W3, a2) + b3
    a3 = activation(z3)
    z4 = np.dot(W4, a3) + b4
    a4 = activation(z4)
    delta4 = hadamart(a4 - y_pt, der_activation(z4))
    delta3 = hadamart(der_activation(z3), np.dot(W4.T, delta4))
    delta2 = hadamart(der_activation(z2), np.dot(W3.T, delta3))
    # Now that all the delta's are known, arrange them
    grad_W2 = np.zeros((2, 2))
    grad_W3 = np.zeros((3, 2))
    grad_W4 = np.zeros((2, 3))
    for j in range(2):
        for k in range(2):
            grad_W2[j,k] = delta2[j]*x_pt[k]
    for j in range(3):
        for k in range(2):
            grad_W3[j,k] = delta3[j]*a2[k]
    for j in range(2):
        for k in range(3):
            grad_W4[j,k] = delta4[j]*a3[k]

    return np.concatenate((delta2, delta3, delta4, grad_W2, 
                            grad_W3, grad_W4), axis=None)


# Total gradient of the cost function; average of the single one
def grad_cost (x, y, b2, b3, b4, W2, W3, W4, N=10):
    result = np.zeros(23)
    for i in range(N):
        result = result + ith_grad_cost(x[i], y[i], b2, b3, b4, W2, W3, W4)
    return result / N


#### GRADIENT DESCENT FUNCTION. It does NOT globally modify the params ###
def performGradientDescent(x, y, b2, b3, b4, W2, W3, W4, 
                                steps = 10000, eta = 10.0, mode_tol=0.001):
    # Backup the starting parameters to compute the distance at the end
    ob2 = np.copy(b2)
    ob3 = np.copy(b3)
    ob4 = np.copy(b4)
    oW2 = np.copy(W2)
    oW3 = np.copy(W3)
    oW4 = np.copy(W4)
    print("Starting cost: ", total_cost (x, y, b2, b3, b4, W2, W3, W4))
    for i in range(steps):
#     print("Step ", i, "cost", total_cost (x, y, b2, b3, b4, W2, W3, W4))
        result = grad_cost(x, y, b2, b3, b4, W2, W3, W4)
        b2 = b2 - eta*result[0:2]
        b3 = b3 - eta*result[2:5]
        b4 = b4 - eta*result[5:7]
        W2 = W2 - eta*(result[7:11].reshape(2, 2))
        W3 = W3 - eta*(result[11:17].reshape(3, 2))
        W4 = W4 - eta*(result[17:23].reshape(2, 3))

    print("Final cost after", steps, "steps:",
            "{:.7e}".format(total_cost (x, y, b2, b3, b4, W2, W3, W4)))
    cost_diff = np.fabs(total_cost(x, y, ob2, ob3, ob4, oW2, oW3, oW4) - \
                                    total_cost(x, y, b2, b3, b4, W2, W3, W4))
    print("Energy difference (the lower, the more likely it's a mode): ",
            "{:.7e}".format(cost_diff))
    if cost_diff < mode_tol:
        return True, b2, b3, b4, W2, W3, W4
    else:
        return False, b2, b3, b4, W2, W3, W4


# Given an array of 23 numbers, split it into the NN variables
def split_params(params, verbose = False):
    b2 = params[0:2]
    b3 = params[2:5]
    b4 = params[5:7]
    W2 = params[7:11].reshape(2, 2)
    W3 = params[11:17].reshape(3, 2)
    W4 = params[17:23].reshape(2, 3)
    if verbose:
        print("Parameters split into:")
        print("b2 = ", b2)
        print("b3 = ", b3)
        print("b4 = ", b4)
        print("W2 = ", W2)
        print("W3 = ", W3)
        print("W4 = ", W4)
    return b2, b3, b4, W2, W3, W4
###########################################################

# Plot the contour of the Neural Newtwork
def nn_contour_plot(x,y,b2, b3, b4, W2, W3, W4, ptnum = -1):
    # Visualizing the random datapoints x
    x1x=[]
    x1y=[]
    x2x=[]
    x2y=[]
    for i in range(N):
        if y[i][0] == 1:
            x1x.append(x[i][0])
            x1y.append(x[i][1])
        else:
            x2x.append(x[i][0])
            x2y.append(x[i][1])
#    plt.scatter(x2x, x2y, c='blue', marker='x')
#    plt.scatter(x1x, x1y, c='red')
    # Contour plot
    xmsh = np.arange(0, 1, 0.025)
    ymsh = np.arange(0, 1, 0.025)
    X, Y = np.meshgrid(xmsh, ymsh)
    Z = np.zeros(len(xmsh) * len(ymsh))
    Z = Z.reshape(len(xmsh), len(ymsh))
    for i in range(len(xmsh)):
        for j in range(len(ymsh)):
            Z[i][j] = NN_to_scalar(NN(np.array([xmsh[i], ymsh[j]]), 
                    b2, b3, b4, W2, W3, W4))
#    fig,ax=plt.subplots(1,1)
    # BE CAREFUL WITH THE axis order!!!!
#    cp = ax.contourf(Y, X, Z) #, colors=["white", "black"])
    plt.contourf(Y, X, Z)#, colors=['green', 'yellow', 'black'])
    # fig.colorbar(cp)
    # ax.set_title(title)
    plt.scatter(x2x, x2y, c='blue', marker='x')
    plt.scatter(x1x, x1y, c='red')
    # Add the information of the current energy level (the lower, the better)
    plot_title ="Cost: " + \
            str("{:.7e}".format(total_cost(x,y,b2,b3,b4,W2,W3,W4)))
    # Ptnum "point number", used when plotting the modes
    if (ptnum != -1):
        plot_title = "mode #"\
                + str(ptnum) + '; ' + plot_title
    plt.title (plot_title)
#    plt.show()

# Global variables needes for defining the optimization's potential
N = 10
x = np.array([[0.1, 0.1], [0.3, 0.4], [0.1, 0.5], [0.6, 0.9], [0.4, 0.2], 
                [0.6, 0.3], [0.5, 0.6], [0.9, 0.2], [0.4, 0.4], [0.7, 0.6]])
y = np.array([[1,0], [1,0], [1,0], [1,0], [1,0], [0,1], [0,1], [0,1], [0,1],
                    [0,1]])

###---------------------------------------------------------------------#

if __name__ == "__main__":
    print("Random points\n:")
    print(x)
    print("Their classification:\n")
    print(y)
    # Visualize the random data
    x1x=[]
    x1y=[]
    x2x=[]
    x2y=[]
    for i in range(N):
        if y[i][0] == 1:
            x1x.append(x[i][0])
            x1y.append(x[i][1])
        else:
            x2x.append(x[i][0])
            x2y.append(x[i][1])
   
    # Parameters for the NN
    params_together = np.random.random_sample(23) * 0.5 
    b2, b3, b4, W2, W3, W4 = split_params(params_together)
    print("BEFORE optimization")
    nn_contour_plot(x, y, b2, b3, b4, W2, W3, W4)
    plt.show()
    #print("#point VS original class VS NN class")
    #for i in range(N):
    #    print(i, y[i][0], NN_to_scalar(NN(x[i], b2, b3, b4, W2, W3, W4)))
    
    #print("Checking for non zeroes...")
    #for i in np.arange(0, 1, 0.025):
    #    for j in np.arange(0, 1, 0.025):
    #        tmp = NN_to_scalar(NN(np.array([i,j]), b2, b3, b4, W2, W3, W4))
    #        if (tmp == 1):
    #            print("NN(", i, ",", j, "=", tmp)
    
    
    _,b2,b3,b4,W2,W3,W4 = performGradientDescent(x, y, b2, b3, b4, W2, W3, W4,
            1000000, 0.05)
    print("AFTER gradient descent")
    nn_contour_plot(x, y, b2, b3, b4, W2, W3, W4)
    plt.show()
    
    #print("#point VS original class VS NN class")
    #for i in range(N):
    #    print(i, y[i][0], NN_to_scalar(NN(x[i], b2, b3, b4, W2, W3, W4)))
    

    
