import numpy as np

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    dz = np.array(z, copy=True)
    dz[z <= 0] = 0
    dz[z > 0] = 1
    return dz

def linear(z):
    return z

def linear_derivative(z):
    return np.ones_like(z)

def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(z) ** 2

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def activation_functions():
    return {
        'relu': (relu, relu_derivative),
        'linear': (linear, linear_derivative),
        'tanh': (tanh, tanh_derivative),
        'sigmoid': (sigmoid, sigmoid_derivative)
    }