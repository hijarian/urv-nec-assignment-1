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
    activation_function (string): Activation function to be used in the network. Possible values: 'relu', 'linear'.
    validation_split (float): Fraction of data to be used for validation.

    xi (list): List containing the output of each layer.
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
  def __init__(self, layers, epochs, learning_rate, momentum, activation_function, validation_split, random_seed=None):
    self.L = len(layers)
    self.n = layers.copy()
    self.epochs = epochs
    self.learning_rate = learning_rate
    self.momentum = momentum
    self.activation_function = activation_function
    self.validation_split = validation_split

    self.xi = []
    for layer in range(self.L):
      self.xi.append(np.zeros(layers[layer]))

    # classical solution is to zero the values out, but it's kind of redundant
    # as we are going to randomize them in the fit method anyway
    self.w = []
    # placeholder for the zeroth layer
    self.w.append(np.zeros((1, 1)))
    for layer in range(1, self.L):
      self.w.append(np.zeros((layers[layer], layers[layer - 1])))

    self.generator = np.random.default_rng(random_seed)

    # We collect the errors for each epoch in these arrays
    # so we can plot them later
    self.train_errors = []
    self.validation_errors = []

  # Scale input and/or output patterns!
  def fit(self, x: np.ndarray, y: np.ndarray) -> None:
    # Validate dimensions of x and y
    if x.shape[0] != y.shape[0]:
      raise ValueError("The number of samples in x and y must be equal.")
    if x.shape[1] != self.n[0]:
      raise ValueError(f"The number of features in x must be {self.n[0]}.")

    # Split the data into training and validation sets
    val_size = int(len(x) * self.validation_split)
    x_train, x_val = x[val_size:], x[:val_size]
    y_train, y_val = y[val_size:], y[:val_size]

    # Initialize all weights and thresholds randomly
    for lay in range(1, self.L):
      self.w[lay] = self.generator.standard_normal(self.n[lay], self.n[lay - 1])

    # Training loop BEGIN

    # reset the training and validation errors arrays cache as we are training again
    # these are stored in the instance so we can access them later by calling self.loss_epochs()
    self.train_errors = []
    self.validation_errors = []

    for epoch in range(self.epochs):
      # shuffle the patterns in x and y synchronously
      indices = self.generator.permutation(len(x_train))
      x_shuffled = x_train[indices]
      y_shuffled = y_train[indices]
      
      train_error = 0

      # then iterate over the x and y patterns
      for pattern_index in range(len(x_shuffled)):
        # Choose the pattern (xµ, zµ) of the shuffled training set
        x_mu = x_shuffled[pattern_index]
        y_mu = y_shuffled[pattern_index]

        # Feed-forward propagation of pattern xµ to obtain the output o(xµ)
        self.forward(x_mu)

        # Back-propagate the error for this pattern
        self.backward(y_mu)

        # Update the weights and thresholds
        self.update_weights()

      # Feed-forward all training patterns and calculate their prediction quadratic error
      current_epoch_prediction = self.predict(x_train)
      train_error = 0
      for i in range(len(x_train)):
        train_error += np.sum((current_epoch_prediction[i] - y_train[i]) ** 2)
      train_error /= len(x_train)

      # Feed-forward all validation patterns and calculate their prediction quadratic error
      current_epoch_validation_prediction = self.predict(x_val)
      val_error = 0
      for i in range(len(x_val)):
        val_error += np.sum((current_epoch_validation_prediction - y_val[i]) ** 2)
      val_error /= len(x_val)

      # Optional: Print the evolution of the training and validation errors
      print(f"Epoch {epoch + 1}/{self.epochs}, Training Error: {train_error}, Validation Error: {val_error}")



  def predict(self, x):
    y = []
    for i in range(len(x)):
      self.forward(x[i])
      y.append(self.xi[self.L - 1])
    return y
  
  """
  returns 2 arrays of size (n_epochs, 2) that
contain the evolution of the training error and the validation error for each of
the epochs of the system, so this information can be plotted.
  """
  def loss_epochs(self):
    return np.array(self.train_errors), np.array(self.val_errors)

  def forward(self, x):
    self.xi[0] = x
    for lay in range(1, self.L):
        z = np.dot(self.w[lay], self.xi[lay - 1])
        if self.activation_function == 'relu':
            self.xi[lay] = np.maximum(0, z)
        elif self.activation_function == 'linear':
            self.xi[lay] = z
        # todo: Add other activation functions as needed

  def backward(self, y):
    error = self.xi[self.L - 1] - y
    deltas = [None] * self.L
    deltas[self.L - 1] = error

    for lay in range(self.L - 2, 0, -1):
        if self.activation_function == 'relu':
            delta = np.dot(self.w[lay + 1].T, deltas[lay + 1])
            delta[self.xi[lay] <= 0] = 0
            deltas[lay] = delta
        # Add other activation functions as needed
  
  def update_weights(self):
    for lay in range(1, self.L):
      self.w[lay] -= self.learning_rate * self.dW[lay]
