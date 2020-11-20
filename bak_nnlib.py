# This is a very simple library to generate naive Neural Networks
import numpy as np
from numpy import log, exp
import matplotlib.pyplot as plt

def ber():
    if np.random.uniform() < 0.5:
        return 0.
    return 1.

# Global variables that defines my Newtwork on which to experiment
NUM_INPUTS = 2
NUM_HIDDEN_LAYERS = 3
NUM_NODES_HIDDEN = [4, 4, 2]
NUM_NODES_OUTPUT = 1
global_N_points = 10
old_seed = np.random.get_state()
np.random.seed(3)
global_X = np.random.uniform(size = [global_N_points, 2])
global_y = np.array([ber() for i in range(global_N_points)])
np.random.set_state(old_seed)

def plot_data():
    X1 = []
    X0 = []
    for i in range(len(global_y)):
        if global_y[i] == 1:
            X1.append(global_X[i])
        else:
            X0.append(global_X[i])
    X1 = np.asanyarray(X1)
    X0 = np.asanyarray(X0)
    plt.scatter(X0[:, 0], X0[:, 1], color = 'red')
    plt.scatter(X1[:, 0], X1[:, 1], color = 'blue')
    plt.title("Data to classify")
    plt.show()


def get_num_params():
    tot = NUM_INPUTS * NUM_NODES_HIDDEN[0] + NUM_NODES_HIDDEN[0]
    for i in range(1, NUM_HIDDEN_LAYERS + 1):
        if (i == NUM_HIDDEN_LAYERS):
            tot += NUM_NODES_HIDDEN[i-1] * NUM_NODES_OUTPUT + NUM_NODES_OUTPUT
        else:
            tot += NUM_NODES_HIDDEN[i-1] * NUM_NODES_HIDDEN[i] + \
                   NUM_NODES_HIDDEN[i]
    print("This model requires ", tot, "parameters")
    return tot

# Step 1: a function that, given the neural network structure and an array of the right
# lenghts, create a NN with parameters as in the array
def init_network(params, num_inputs, num_hidden_layers, 
                                    num_nodes_hidden, num_nodes_output):

    num_nodes_previous = num_inputs # number of nodes in the previous layer
    network = {}
    offset = 0

    # DEBUG function: just compute the number of parameters
    num_params = num_inputs * num_nodes_hidden[0] + num_nodes_hidden[0]
    for i in range(1, num_hidden_layers + 1): 
        if (i == num_hidden_layers):
            num_params += num_nodes_hidden[i-1] * num_nodes_output + num_nodes_output
        else:
            num_params += num_nodes_hidden[i-1] * num_nodes_hidden[i] + num_nodes_hidden[i]
#    print("Total number of network parameters: ", num_params)

    if (len(params) != num_params):
        print("ERROR: wrong parameters dimension")
    if (num_hidden_layers != len(num_nodes_hidden)):
        print("ERROR: wrong input")
    
    # loop through each layer and initialize the weights and biases associated with each layer
    for layer in range(num_hidden_layers + 1):
        # Start by giving names to each layer
        if layer == num_hidden_layers:
            layer_name = 'output' # name last layer in the network output
            num_nodes = num_nodes_output
        else:
            layer_name = 'layer_{}'.format(layer + 1)
            num_nodes = num_nodes_hidden[layer]

        # initialize weights and bias for each node
        network[layer_name] = {}
        for node in range(num_nodes):
            node_name = 'node_{}'.format(node+1)
            network[layer_name][node_name] = {
                'weights': np.asanyarray(params[offset:offset+num_nodes_previous]),
                'bias'   : np.asanyarray(params[offset + num_nodes_previous])
            }
            offset = offset + num_nodes_previous + 1
            #    'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
            #    'bias': np.around(np.random.uniform(size=1), decimals=2),
            #}
        num_nodes_previous = num_nodes

    return network # return the network


### DEBUG variable
p = [i for i in range(1, 60)]
small_network2 = init_network(p, 2, 3, [2, 2, 2], 1)
small_network = init_network(p, 5, 3, [3, 2, 3], 1)
small_network_OK = init_network(p, 2, 3, [3, 3, 2], 1)

### Step 2: given an input, evaluate a network


def compute_weighted_sum(inputs, weights, bias):
    return np.sum(inputs * weights) + bias

def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))

# Define the ReLU function
def node_activation(weighted_sum):
    return sigmoid(weighted_sum)
#    return max(0., weighted_sum)


def forward_propagate(network, inputs):

    layer_inputs = list(inputs) # start with the input layer as the input to the first hidden layer

    for layer in network:

        layer_data = network[layer]

        layer_outputs = []
        for layer_node in layer_data:

            node_data = layer_data[layer_node]

            # compute the weighted sum and the output of each node at the same time
            node_output = node_activation(compute_weighted_sum(layer_inputs, node_data['weights'], node_data['bias']))
#            layer_outputs.append(np.around(node_output[0], decimals=4))
            layer_outputs.append(node_output)

#        if layer != 'output':
#            print('The outputs of the nodes in hidden layer number {} is {}'.format(layer.split('_')[1], layer_outputs))

        layer_inputs = layer_outputs # set the output of this layer to be the input to next layer

#    network_predictions = sigmoid(layer_outputs[0])
    network_predictions = layer_outputs[0]
    return network_predictions

# STEP 2: some good datapoint
def into01(y):
    ypred = np.copy(y)
    for i in range(len(ypred)):
        if ypred[i] >= 1. :
            ypred[i] = 0.99
        elif ypred[i] <= 0.:
            ypred[i] = 0.01
    return ypred

def binary_cross(ytrue, ypred):
    n = len(ytrue)
    ypred = into01(ypred)
    sm = 0.
    for i in range(n):
        if (ypred[i] > 0. and ypred[i] < 1.):
            sm += ytrue[i] * log(ypred[i]) + \
                    (1. - ytrue[i]) * log(1. - ypred[i])
        else:
            input("Cross entropy error")
    return (- sm / n) * 100.


def l2square(ytrue, ypred):
    n = len(ytrue)
    sm = 0.
    for i in range(n):
        sm += (ytrue[i] - ypred[i]) ** 2
    return (np.sqrt(sm) / n) * 100


# STEP 3: define the loss function
# THE LOSS AND ACCURACY  FUNCTION MUST BE THE ONLY 
# ONE DEPENDING ON GLOBAL VARIABLES
def loss(p):
    # Create a Network
    #loc_nn = init_network(p, 2, 3, [3, 3, 2], 1)
    loc_nn = init_network(p, NUM_INPUTS, NUM_HIDDEN_LAYERS, 
                                        NUM_NODES_HIDDEN, NUM_NODES_OUTPUT)
    # Evaluate each datapoint on the created newtork
    y_hat = np.array([forward_propagate(loc_nn, global_X[i]) \
            for i in range(global_N_points)])
#    print("True: \t\t", global_y)
#    print("Predicted: \t", y_hat)
#    input("OK?")
    # Binary entropy loss function ?
    return l2square(global_y, y_hat)
#    return binary_cross(global_y, y_hat)




def from_prob_to_01(yyy):
    yy = np.copy(yyy)
    for i in range(len(yy)):
        if yy[i] < 0.5:
            yy[i] = 0.
        else:
            yy[i] = 1.
    return yy


def accuracy(p):
    loc_nn = init_network(p, NUM_INPUTS, NUM_HIDDEN_LAYERS,
                                        NUM_NODES_HIDDEN, NUM_NODES_OUTPUT)
    # Evaluate each datapoint on the created newtork
    y_hat = np.array([forward_propagate(loc_nn, global_X[i])\
            for i in range(global_N_points)])
    y_hat = from_prob_to_01(y_hat)
    correct = 0
    for i in range(len(y_hat)):
        if (global_y[i] == y_hat[i]):
            correct += 1
    return correct * 100 / len(y_hat)


def from_R_to_prob(yy):
    yyy = np.copy(yy)
    for i in range(len(yyy)):
        yyy[i] = sigmoid(yyy[i])
    return yyy

def accuracy_with_l2err(p):
 # Create a Network
    loc_nn = init_network(p, NUM_INPUTS, NUM_HIDDEN_LAYERS,
                                        NUM_NODES_HIDDEN, NUM_NODES_OUTPUT)
    # Evaluate each datapoint on the created newtork
    y_hat = np.array([forward_propagate(loc_nn, global_X[i]) \
            for i in range(global_N_points)])
    print("Initial y predict: ", y_hat)
    y_hat = from_R_to_prob(y_hat)
    print("To [0,1]: ", y_hat)
    y_hat = from_prob_to_01(y_hat)
    print("to labels:: ", y_hat)
    print("True labels: ", global_y)
    correct = 0
    for i in range(len(y_hat)):
        if (global_y[i] == y_hat[i]):
            correct += 1
    return correct * 100 / len(y_hat)

