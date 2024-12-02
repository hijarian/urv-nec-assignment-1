import numpy as np

class NeuralNet:
  """
  NeuralNet class for creating and training a neural network.
  Attributes:
    L (int): Number of layers in the neural network.
    n (list): List containing the number of neurons in each layer.
    epochs (int): Number of epochs for training.
    learning_rate (float): Learning rate for weight updates.
    momentum (float): Momentum factor for weight updates.
    activation_function (callable): Activation function to be used in the network.
    validation_split (float): Fraction of data to be used for validation.
    xi (list): List of numpy arrays representing the activations of each layer.
    w (list): List of numpy arrays representing the weights between layers.
  Methods:
    fit(x, y):
      Trains the neural network on the provided data.
      Args:
        x (numpy.ndarray): Input data of shape (n_samples, n_features).
        y (numpy.ndarray): Target values of shape (n_samples,).
    predict(x):
      Predicts the output for the given input data.
      Args:
        x (numpy.ndarray): Input data of shape (n_samples, n_features).
      Returns:
        list: Predicted values for all input samples.
    loss_epochs():
      Placeholder method for calculating loss over epochs.
    forward(x):
      Placeholder method for forward propagation.
      Args:
        x (numpy.ndarray): Input data for a single sample.
    backward(y):
      Placeholder method for backward propagation.
      Args:
        y (numpy.ndarray): Target value for a single sample.
    update_weights():
      Placeholder method for updating weights.
  """
  def __init__(self, layers, epochs, learning_rate, momentum, activation_function, validation_split):
    self.L = len(layers)
    self.n = layers.copy()
    self.epochs = epochs
    self.learning_rate = learning_rate
    self.momentum = momentum
    self.activation_function = activation_function
    self.validation_split = validation_split

    self.xi = []
    for lay in range(self.L):
      self.xi.append(np.zeros(layers[lay]))

    self.w = []
    self.w.append(np.zeros((1, 1)))
    for lay in range(1, self.L):
      self.w.append(np.zeros((layers[lay], layers[lay - 1])))

  def fit(self, x, y):
    for epoch in range(self.epochs):
      for i in range(len(x)):
        self.forward(x[i])
        self.backward(y[i])
        self.update_weights()


  def predict(self, x):
    y = []
    for i in range(len(x)):
      self.forward(x[i])
      y.append(self.xi[self.L - 1])
    return y
  
  def loss_epochs():
    pass # todo

  def forward(self, x):
    pass # todo

  def backward(self, y):
    pass # todo
  
  def update_weights(self):
    pass # todo