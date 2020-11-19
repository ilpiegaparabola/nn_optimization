# This is a first example where we try to use my Monte Carlo exploration
# method to train the simplest realistic convolutional NN by using
# the Keras interface. If my method wants any chance to work in practice,
# it needs to be successfull with the basic "hello worlds" methods.
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from numpy.linalg import norm
import numpy as np

# Mean Squared Error function
mse = keras.losses.MeanSquaredError()

def l2err(v1, v2):
    sm = 0.
    for i in range(len(v1)):
        sm += (v1[i] - v2[i]) ** 2
    return sm / v1.shape[0]

# Taking the data from the tutorial on regression
data = pd.read_csv('concrete_data.csv')
concrete_data_columns = data.columns

# all columns except Strength
X = data[concrete_data_columns[concrete_data_columns != 'Strength']]
y = data['Strength'] # Strength column
# Normalizing the data
X = (X - X.mean()) / X.std()

# For the moment, use all of them for training - we are
# first of all interesting in understanding if there are chanches of
# a working optimization
X_train = np.copy(X)
y_train = np.copy(y)

n_cols = X_train.shape[1]

# NOTA BENE! Una volta che x e y sono fissati, e non importa come,
# conta solo che lo siano, allora la funzione costo si limita ad essere
# una ordinaria da R^p a R, quindi in principio ottimizzabile
# usando MC.

# define regression model
def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(20, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(10, activation='relu'))
#    model.add(Dense(30, activation='relu'))
#    model.add(Dense(30, activation='relu'))
    model.add(Dense(1))

    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model = regression_model()

if __name__ == "__main__":
    model.fit(X_train, y_train, validation_split=0.3, epochs=2000, verbose=2)
    print("Ok, model fitted")
    y_train_hat = model.predict(X_train).reshape(y_train.shape)
#    y_train_hat = y_train_hat.reshape(y_train.shape)
    print("Keras MSE error: ", mse(y_train, y_train_hat).numpy()) 
    print("Relative training error: ", 
            int(100. * norm(y_train - y_train_hat) / norm(y_train)), "%")
    print("The model has ", model.count_params(), "parameters")


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
#    y_train_hat = model.predict(X_train)
    y_train_hat = model.predict(X_train).reshape(y_train.shape)
    mse = keras.losses.MeanSquaredError()
    return mse(y_train, y_train_hat).numpy() * 100


# In my case, for this simple NN, the number of parameters
# is 3051. Nota che NON dipende dalla funzione costo, eh.
NN_dimension = model.count_params()
myparams = np.random.uniform(0, 1, NN_dimension)
for_keras = params_to_keras(myparams)
print("Just an evaluation with random coefficients: ")
print(keras_cost(myparams))


#### FANTASTICO!!!! Dunque, allora adesso ho PERFETTAMENTE la mia
# funzione potenziale!!
# Now, time to implement the Monte Carlo approach!
