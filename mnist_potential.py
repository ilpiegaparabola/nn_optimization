# This is a first example where we try to use my Monte Carlo exploration
# method to train the simplest realistic convolutional NN by using
# the Keras interface. If my method wants any chance to work in practice,
# it needs to be successfull with the basic "hello worlds" methods.
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.datasets import mnist
import tensorflow as tf
from numpy.linalg import norm
import numpy as np

np.random.seed(0)

(X_train, y_train) , (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

X_train = X_train / 255
X_test = X_test / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = y_test.shape[1]

# NOTA BENE! Una volta che x e y sono fissati, e non importa come,
# conta solo che lo siano, allora la funzione costo si limita ad essere
# una ordinaria da R^p a R, quindi in principio ottimizzabile
# usando MC.

def convolutional_model():
    model = Sequential()
    model.add(Conv2D(16, (5, 5), strides=(1, 1), activation='relu', 
        input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    ##model.compile(optimizer='adam', loss='categorical_crossentropy', 
   #  metrics=['accuracy'])
    model.compile(optimizer='sgd', loss = tf.keras.losses.MeanSquaredError())
    return model

model = convolutional_model()
#model.fit(X_train, y_train, validation_data = (X_test, y_test),
#        epochs = 100, batch_size = 200, verbose = 2)

# A simple function to convert the probabilistic interpretation
# of softmax into a 0-1 encoding
def from_prob_to_01(yy):
    for point in yy:
        dim = len(point)
        best = max(point)
        for i in range(dim):
            if point[i] < best:
                point[i] = 0
            else:
                point[i] = 1
    return yy

# Evaluating the model accuracy:
#y_train_hat = from_prob_to_01(model.predict(X_train))
#y_test_hat = from_prob_to_01(model.predict(X_test))
#print("Training error: ",
#        100. * norm(y_train_hat - y_train)/ norm(y_train))
#print("Test error: ", 100. * norm(y_test_hat - y_test)/ norm(y_test))


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
    y_train_hat = from_prob_to_01(model.predict(X_train))
    mse = keras.losses.MeanSquaredError()
    return mse(y_train, y_train_hat).numpy() * 100

#y_train_hat = from_prob_to_01(model.predict(X_train))
#y_test_hat = from_prob_to_01(model.predict(X_test))
#print("Training error: ",
#        100. * norm(y_train_hat - y_train)/ norm(y_train))
#print("Test error: ", 100. * norm(y_test_hat - y_test)/ norm(y_test))


# In my case, for this simple NN, the number of parameters
# is 231.926. Nota che NON dipende dalla funzione costo, eh.
NN_dimension = 231926
myparams = np.random.uniform(0, 1, 231926)
for_keras = params_to_keras(myparams)
print(keras_cost(myparams))


#### FANTASTICO!!!! Dunque, allora adesso ho PERFETTAMENTE la mia
# funzione potenziale!!
# Now, time to implement the Monte Carlo approach!
