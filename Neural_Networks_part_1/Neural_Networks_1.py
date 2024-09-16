import numpy as np

def get_weights(input_shape, output_shape):
    # return a weight matrix
    # initialize all weights by o.5

    return 0.5 * np.ones((output_shape, input_shape))

def get_bias(shape):
    # bias vector
    return 0.5*np.ones(shape)

def sigmoid(z: float):
    return 1/(1 + np.exp(-z))

def loss(y, y_pred):
    return np.linalg((y - y_ped), ord = 2)

# the data set
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([1, 0, 0, 1])

"""
Feed data through network
"""

def feed_forward(x0):
    # first hidden layer
    z1 = np.dot(get_weights(2, 2), x0) + get_bias(2)
    x1 = sigmoid(z1)

    # second hidden layer
    z2 = np.dot(get_weights(2, 2), x1) + get_bias(2)
    x2 = sigmoid(z2)
    print(f'z^2 = {z2}')
    print(f'x^2 = {x2}\n')

    # output layer
    z3 = np.dot(get_weights(2, 1), x2) + get_bias(1)
    x3 = sigmoid(z3)
    print(f'z^3 = {z3}')
    print(f'x^3 = {x3}\n')

# compute forward pass for each example
for x in X:
    print(f'\n----- sample {x}')
    feed_forward(x)

"""
Tensor implementation
"""
# Initialize input data(multiple 2-dimensional vectors as a batch)
input_data = X

# Initialize weights and biases fot the layers
input_size = 2
hidden1_size = 2
hidden2_size = 2
output_size = 1

# Randomly initialize weights and biases
weights_hidden1 = np.ones((input_size, hidden1_size)) * 0.5
biases_hidden1 = np.ones((1, hidden1_size)) * 0.5

weights_hidden2 = np.ones((hidden1_size, hidden2_size)) * 0.5
biases_hidden2 = np.ones((1, hidden2_size)) * 0.5

weights_output = np.ones((hidden2_size, output_size)) * 0.5
biases_output = np.ones((1, output_size)) * 0.5

# Forward pass through the network
hidden1_output = sigmoid(np.dot(input_data, weights_hidden1) + biases_hidden1)
hidden2_output = sigmoid(np.dot(hidden1_output, weights_hidden2) + biases_hidden2)
output = sigmoid(np.dot(hidden2_output, weights_output) + biases_output)

print("Input data:")
print(input_data)
print("\nOutput:")
print(output)