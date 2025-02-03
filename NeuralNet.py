import numpy as np
import matplotlib.pyplot as plt
from activation_functions import activation_functions

class NeuralNet:
  """
  NeuralNet class for creating and training a neural network.
  Attributes:
    L (int): Number of layers in the neural network.
    n (list): List containing the number of neurons in each layer.
    epochs (int): Number of epochs for training.
    learning_rate (float): Learning rate for weight updates.
    momentum (float): Momentum factor for weight updates.
    activation_function (string): Activation function to be used in the network. Possible values: 'relu', 'linear', 'tanh', 'sigmoid'.
    validation_split (float): Fraction of data to be used for validation.

    xi (list): List containing the output of each layer.
    w (list): List of numpy arrays representing the weights between layers.
    theta (list): List of numpy arrays representing the thresholds (biases) for each layer.

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
  def __init__(self, *, layers, epochs=100, learning_rate=1.1, momentum=0.9, activation_function_name='sigmoid', validation_split=0.2, random_seed=None, visualize=True):
    self.L = len(layers)
    self.n = layers.copy()
    self.epochs = epochs
    self.learning_rate = learning_rate
    self.momentum = momentum
    self.validation_split = validation_split
    self.visualize = visualize

    self.activation_function_name = activation_function_name

    [self.activation_function, self.activation_function_derivative] = activation_functions()[activation_function_name]
    if (self.activation_function is None) or (self.activation_function_derivative is None):
      raise ValueError(f"Unsupported activation function: {activation_function_name}")

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

    self.theta = []
    self.theta.append(np.zeros(1))
    for layer in range(1, self.L):
      self.theta.append(np.zeros(layers[layer]))

    self.generator = np.random.default_rng(random_seed)

    # We collect the errors for each epoch in these arrays
    # so we can plot them later
    self.training_errors = []
    self.validation_errors = []

    # Initialize velocity terms for momentum
    self.v = []
    self.v.append(np.zeros((1, 1)))
    for layer in range(1, self.L):
      self.v.append(np.zeros((layers[layer], layers[layer - 1])))

    # Initialize gradients
    self.d_w = []
    self.d_w.append(np.zeros((1, 1)))
    for layer in range(1, self.L):
      self.d_w.append(np.zeros((layers[layer], layers[layer - 1])))

    self.d_theta = []
    self.d_theta.append(np.zeros(1))
    for layer in range(1, self.L):
      self.d_theta.append(np.zeros(layers[layer]))

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
      prev_layer_size = self.n[lay - 1]
      # poor man's Xavier initialization
      self.w[lay] = self.generator.normal(0, 1 / prev_layer_size, size=(self.n[lay], self.n[lay - 1]))
      self.theta[lay] = np.zeros(self.n[lay])

    # Training loop BEGIN

    # reset the training and validation errors arrays cache as we are training again
    # these are stored in the instance so we can access them later by calling self.loss_epochs()
    self.training_errors = []
    self.validation_errors = []

    if self.visualize:
      plt.ion()
      figure, axs = plt.subplots(5, self.L)
      images_weight = [None] * self.L
      plots_theta = [None] * self.L
      images_delta_weights = [None] * self.L
      plots_delta_thetas = [None] * self.L
      for layer in range(1, self.L):
        column = layer - 1
        axs[0, column].clear()
        images_weight[layer] = axs[0, column].matshow(self.w[layer], cmap='hot', interpolation='nearest')
        axs[0, column].set_title(f'Layer {layer} weights')
        axs[0, column].axis('off')
        figure.colorbar(images_weight[layer], ax=axs[0, column])

        axs[1, column].clear()
        plots_theta[layer] = axs[1, column].plot(self.theta[layer])
        axs[1, column].set_title(f'Layer {layer} theta')

        axs[2, column].clear()
        images_delta_weights[layer] = axs[2, column].matshow(self.d_w[layer], cmap='hot', interpolation='nearest')
        axs[2, column].set_title(f'Layer {layer} weights diff')
        axs[2, column].axis('off')
        figure.colorbar(images_delta_weights[layer], ax=axs[2, column])

        axs[3, column].clear()
        plots_delta_thetas[layer] = axs[3, column].plot(self.d_theta[layer])
        axs[3, column].set_title(f'Layer {layer} theta diff')

      axs[4, 0].set_title('Training Error')
      axs[4, 1].set_title('Validation Error')

      plt.show(block=False)
      figure.canvas.draw_idle()
      figure.canvas.flush_events()

    for epoch in range(self.epochs):
      # shuffle the patterns in x and y synchronously
      indices = self.generator.permutation(len(x_train))
      x_shuffled = x_train[indices]
      y_shuffled = y_train[indices]
      
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

        if self.visualize and pattern_index % self.visualize == 0:
          for layer in range(1, self.L):
            column = layer - 1;
            # set array simply silently does not do anything
            # images_weight[layer].set_array(self.w[layer])

            # so I recreate the image every time
            axs[0, column].clear()
            images_weight[layer] = axs[0, column].matshow(self.w[layer], cmap='hot', interpolation='nearest')
            axs[0, column].axis('off')

            axs[1, column].clear()
            axs[1, column].plot(self.theta[layer])

            # set array simply silently does not do anything
            # images_delta_weights[layer].set_array(self.d_w[layer])

            # so I recreate the image every time
            axs[2, column].clear()
            images_delta_weights[column] = axs[2, column].matshow(self.d_w[layer], cmap='hot', interpolation='nearest')
            axs[2, column].axis('off')

            axs[3, column].clear()
            axs[3, column].plot(self.d_theta[layer])

          # plt.show(block=False)
          figure.canvas.draw_idle()
          figure.canvas.flush_events()

      train_error = self.calculate_error(x_train, y_train)
      self.training_errors.append(train_error)

      validation_error = self.calculate_error(x_val, y_val)
      self.validation_errors.append(validation_error)

      # Optional: Print the evolution of the training and validation errors
      if self.visualize:
        for layer in range(1, self.L):
          column = layer - 1
          # set array simply silently does not do anything
          # images_weight[layer].set_array(self.w[layer])

          # so I recreate the image every time
          axs[0, column].clear()
          images_weight[layer] = axs[0, column].matshow(self.w[layer], cmap='hot', interpolation='nearest')
          axs[0, column].axis('off')

          axs[1, column].clear()
          axs[1, column].plot(self.theta[layer])

          # set array simply silently does not do anything
          # images_delta_weights[layer].set_array(self.d_w[layer])

          # so I recreate the image every time
          axs[2, column].clear()
          images_delta_weights[layer] = axs[2, column].matshow(self.d_w[layer], cmap='hot', interpolation='nearest')
          axs[2, column].axis('off')

          axs[3, column].clear()
          axs[3, column].plot(self.d_theta[layer])
          
        print(f"Epoch {epoch + 1}/{self.epochs}, Training Error: {train_error}, Validation Error: {validation_error}")
        axs[4, 0].plot(self.training_errors, label='Training Error')
        axs[4, 1].plot(self.validation_errors, label='Validation Error')
        # plt.show(block=False)
        figure.canvas.draw_idle()
        figure.canvas.flush_events()

   # Training loop END

  # Calculate the error for a given set of patterns
  # both x and y are full set of patterns, not a single one
  def calculate_error(self, x: np.ndarray, y: np.ndarray) -> float:
    predictions = self.predict(x)
    error = np.sum((predictions - y) ** 2) / len(x)
    return error

  # Predict the output for a given set of patterns
  def predict(self, x: np.ndarray) -> np.ndarray:
    y = []
    for i in range(len(x)):
      self.forward(x[i])
      y.append(self.xi[-1])
    return np.array(y)
  
  """
  returns 2 arrays of size (n_epochs, 2) that
contain the evolution of the training error and the validation error for each of
the epochs of the system, so this information can be plotted.
  """
  def loss_epochs(self):
    return np.array(self.training_errors), np.array(self.validation_errors)

  # x is a single pattern not the whole training/validation set!
  def forward(self, x):
    self.xi[0] = x
    for lay in range(1, self.L):
        z = np.dot(self.w[lay], self.xi[lay - 1]) - self.theta[lay]
        self.xi[lay] = self.activation_function(z)

  def backward(self, y):
    error = self.xi[-1] - y
    deltas = [None] * self.L
    deltas[-1] = self.activation_function_derivative(self.xi[-1]) * error

    for lay in range(self.L - 2, 0, -1):
      next_layer_weighted_error = np.dot(self.w[lay + 1].T, deltas[lay + 1])
      deltas[lay] = next_layer_weighted_error * self.activation_function_derivative(self.xi[lay])

    # Calculate gradients
    for lay in range(1, self.L):
      # see BP.v2.pdf page 4
      self.d_w[lay]     = self.momentum * self.d_w[lay] - self.learning_rate * np.outer(deltas[lay], self.xi[lay - 1])
      self.d_theta[lay] = self.momentum * self.d_theta[lay] + self.learning_rate * deltas[lay]

  def update_weights(self):
    for lay in range(1, self.L):
      # Update weights
      self.w[lay] += self.d_w[lay]
      # Update thresholds
      self.theta[lay] += self.d_theta[lay]
