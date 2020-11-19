import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from numpy.linalg import norm
from keras.utils import to_categorical
import numpy as np

# Mean Squared Error function
bce = tf.keras.losses.BinaryCrossentropy()

def ber():
    if np.random.uniform() > 0.5:
        return 1
    else:
        return 0

# Generate a random set of N points on the unit square
N_points = 10

old_seed = np.random.get_state()
np.random.seed(1)
X = np.random.uniform(size = [N_points, 2])
y = np.array([ber() for i in range(N_points)])
np.random.set_state(old_seed)
y = to_categorical(y)

n_cols = X.shape[1]
n_classes = 2

# NOTA BENE! Una volta che x e y sono fissati, e non importa come,
# conta solo che lo siano, allora la funzione costo si limita ad essere
# una ordinaria da R^p a R, quindi in principio ottimizzabile
# usando MC.

# define regression model
def classification_model():
    # create model
    model = Sequential()
    model.add(Dense(3, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))
    # compile model
    model.compile(optimizer='adam', loss='binary_crossentropy',
                metrics=['accuracy'])
    return model

model = classification_model()

if __name__ == "__main__":
    model.fit(X, y, validation_split=0, epochs=2000, verbose=2)
    print("Ok, model fitted")
    y_hat = model.predict(X).reshape(y.shape)
    print("Keras LOSS function: ", bce(y_hat, y).numpy())
    print("The model has ", model.count_params(), "parameters")
    model.summary()


##### STARTING NOW WITH MY INTERFACE ####
# Function to get the total element of an array which is linear or matrix
# Useful because each neural layer has biases parameters in the form
# of arrays, and weights in the form of matrices
# but...does it also hold for convolutional networks?
def total_tensor_elements(a):
    tot = 1.
    for i in range(len(a.shape)):
        tot *= a.shape[i]
    return tot

# Given a generic numpy array of the right size, fit it into the
# right format to be used as a parameter for the keras Neural Network
def params_to_keras(p, verbose = False):
    offset = 0
    # Create an array q of the right Keras format
    q = model.get_weights()
    # Loop for setting the parameters for each layer
    for i in range(len(q)):
        take_n = total_tensor_elements(q[i])
        if(verbose):
            print("for layer" , i, "take", take_n, "elements")
        tmp = np.asarray(p[offset:offset+ int(take_n)])
        tmp.shape = q[i].shape
        q[i] = np.copy(tmp)
        offset += int(take_n)
    return q


# Given a numpy array of dimension N, with N equals to the number of the
# NN parameters, compute the model cost
# function according to the keras model
def keras_cost(p):
    p = params_to_keras(p)
    model.set_weights(p)
    y_hat = model.predict(X).reshape(y.shape)
    return bce(y, y_hat).numpy() * 100


# In my case, for this simple NN, the number of parameters
# is 3051. Nota che NON dipende dalla funzione costo, eh.
NN_dimension = model.count_params()
myparams = np.random.uniform(0, 1, NN_dimension)
for_keras = params_to_keras(myparams)
print("Just a LOSS evaluation with random coefficients:", 
                                                int(keras_cost(myparams)))

#### FANTASTICO!!!! Dunque, allora adesso ho PERFETTAMENTE la mia
# funzione potenziale!!
# Now, time to implement the Monte Carlo approach!
