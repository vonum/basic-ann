import numpy as np
import scipy.special

class NeuralNetwork:
  def __init__(self, inodes, hnodes, onodes, alpha):
    self.inodes = inodes
    self.hnodes = hnodes
    self.onodes = onodes
    self.alpha = alpha

    self.initialize_weights()

    self.activation_function = lambda x: scipy.special.expit(x)

  def initialize_weights(self):
    self.wih = np.random.normal(0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
    self.who = np.random.normal(0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

  def train(self, inputs_list, targets_list):
    # turn inputs and targets from 1xn to nx1 dimensions
    targets = np.array(targets_list, ndmin=2).T
    inputs = np.array(inputs_list, ndmin=2).T

    hidden_outputs = self.__calculate_outputs(self.wih, inputs)
    final_outputs = self.__calculate_outputs(self.who, hidden_outputs)

    output_errors = targets - final_outputs
    hidden_errors = np.dot(self.who.T, output_errors)

    # update the weights for the links between the hidden and output layers
    self.who += self.alpha * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), hidden_outputs.T)
    # update the weights for the links between the input and hidden layers
    self.wih += self.alpha * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), inputs.T)

  def predict(self, inputs_list):
    # turn inputs from 1xn to nx1 dimensions
    inputs = np.array(inputs_list, ndmin=2).T

    hidden_outputs = self.__calculate_outputs(self.wih, inputs)
    final_outputs = self.__calculate_outputs(self.who, hidden_outputs)

    return final_outputs

  def __calculate_outputs(self, weights, inputs):
    layer_inputs = np.dot(weights, inputs)
    layer_outputs = self.activation_function(layer_inputs)

    return layer_outputs