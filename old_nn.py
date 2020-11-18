# This is a very simple library to generate naive Neural Networks
import numpy as np
from numpy import log, exp

# Step 1: a function that, given the neural network structure and an array of the right
# lenghts, create a NN with parameters as in the array
def init_network(params, num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output):

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

# Define the ReLU function
def node_activation(weighted_sum):
    return max(0., weighted_sum)


def compute_weighted_sum(inputs, weights, bias):
    return np.sum(inputs * weights) + bias

def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))


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

 #       if layer != 'output':
#            print('The outputs of the nodes in hidden layer number {} is {}'.format(layer.split('_')[1], layer_outputs))

        layer_inputs = layer_outputs # set the output of this layer to be the input to next layer

    network_predictions = sigmoid(layer_outputs[0])
    return network_predictions

def ber():
    if np.random.uniform() < 0.5:
        return 0.
    return 1.

# STEP 2: some good datapoint
global_N_points = 10
old_seed = np.random.get_state()
np.random.seed(1)
global_X = np.random.uniform(size = [global_N_points, 2])
global_y = np.array([ber() for i in range(global_N_points)])
np.random.set_state(old_seed)

just_p = np.random.uniform(-1, 1, 32)

global_NNDIMENSION = 32

# STEP 3: define the loss function
def loss(p):
    # Create a Network
    loc_nn = init_network(p, 2, 3, [3, 3, 2], 1)
    # Evaluate each datapoint on the created newtork
    y_hat = np.array([forward_propagate(loc_nn, global_X[i]) for i in range(global_N_points)])
    # Estimate the difference between y, the real labels, and y_hat, the predicted
    # Binary entropy loss function ?
    sm = - sum([global_y[i] * log(y_hat[i]) + (1 - global_y[i]) * log(1 - y_hat[i]) \
                                                   for i in range(global_N_points)]) / global_N_points
    return sm * 100


def from_prob_to_01(yyy):
    yy = np.copy(yyy)
    for i in range(len(yy)):
        if yy[i] < 0.5:
            yy[i] = 0.
        else:
            yy[i] = 1.
    return yy


def accuracy(p):
     # Create a Network
    loc_nn = init_network(p, 2, 3, [3, 3, 2], 1)
    # Evaluate each datapoint on the created newtork
    y_hat = np.array([forward_propagate(loc_nn, global_X[i]) for i in range(global_N_points)])
    y_hat = from_prob_to_01(y_hat)
    correct = 0
    for i in range(len(y_hat)):
        if (global_y[i] == y_hat[i]):
            correct += 1
    return correct * 100 / len(y_hat)


# Make it ready to be imported 
