import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Softmax

(x_train, y_train), (x_test, y_test) = mnist.load_data()
imagesize = x_train.shape[1]
print(x_train.shape, y_train.shape)

# image_0 = x_train[0]
# plt.imshow(image_0)
# plt.show()

# hyper parameters
n_outputs = 10
n_inputs = imagesize * imagesize   # len(np.unique(y_train)
hidden_layers = [256, 256]
activation_function = 'relu'

x_train = x_train.reshape((-1, imagesize * imagesize))
x_test = x_test.reshape((-1, imagesize * imagesize))

x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

# one_hot encoding on y_train and Y_ test so now the y_train[:5] = 1

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, y_train.shape)
print(y_train[:5])

model = Sequential()
model.add(Dense(units = hidden_layers[0], input_dim = n_inputs, name = 'hidden_0'))
model.add(Activation(activation = activation_function, name = 'relu_0'))
model.add(Dense(units = hidden_layers[1], name = 'hidden_1'))
model.add(Activation(activation = activation_function, name = 'relu_1'))
model.add(Dense(units = n_outputs, name = 'output_layer'))
model.add(Activation(activation = 'softmax', name = 'softmmax'))
print(model.summary())