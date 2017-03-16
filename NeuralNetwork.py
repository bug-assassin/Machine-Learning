import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork(object):
    def __init__(self, sizes, activationFunction, costFunction):
        # sizes = [input_size, middle_size, output_size]

        self.num_layers = len(sizes)
        self.num_hidden_layers = self.num_layers - 2
        self.n = sizes[0]
        self.sizes = sizes
        self.activationFunction = activationFunction
        self.costFunction = costFunction

        self.weights = []
        for i in range(1, self.num_layers):
            self.weights.append(np.random.randn(sizes[i - 1], sizes[i]))

    def feedforward(self, a):
        # if a.shape[1] != self.n:
        #    Exception("Input does not match features length")

        zs = [a]
        activations = [a]
        # a = np.insert(a, 0, 1) # Add bias
        # 0, 1 when num_layers = 3
        for i in range(self.num_layers - 1):
            weights = self.weights[i]
            z = np.dot(a, weights)
            a = self.activationFunction(z)

            zs.append(z)
            activations.append(a)

        return zs, activations

    def backprop(self, x, y, learningRate=0.5):
        """
        :param x: input(s) in the form of a column vector (m x n)
        :param y: Expected output in the form of column vector (m x n)
        :param learningRate:
        :return: nothing
        """
        x = np.array(x)
        zs, activations = self.feedforward(x)
        yHat = activations[-1]

        deltas = np.empty(self.num_layers - 1, dtype=object)

        # last layer delta
        deltas[-1] = np.multiply(self.costFunction(y, yHat, derivative=True), self.activationFunction(zs[-1], derivative=True))

        # all layers delta except last one
        for layer in range(self.num_hidden_layers - 1, -1, -1):
            deltas[layer] = np.multiply(np.dot(deltas[layer + 1], self.weights[layer + 1].T),
                                        self.activationFunction(zs[layer + 1], derivative=True))

        # weights updates (deltas * activation of previous layer)
        for layer in range(self.num_hidden_layers, -1, -1):
            dJW = np.dot(activations[layer].transpose(), deltas[layer])
            self.weights[layer] -= learningRate * dJW


def squaredError(y, yHat, derivative=False):
    if derivative:
        return -(y - yHat)
    else:
        m = y.shape[0]
        J = 0.5 * sum((y - yHat) ** 2) / m
        return J


def sigmoid(z, derivative=False):
    y = 1 / (1 + np.exp(-z))

    if derivative:
        return y * (1 - y)
    return y


def tanh(z, derivative=False):
    y = np.tanh(z)

    if derivative:
        return 1.0 - (y**2)
    return y

class Trainer(object):
    def __init__(self, neuralNetwork):
        np.random.seed(1)  # to be reproducible
        self.nn = neuralNetwork

    def SplitDataset(self, percent_training=60, percent_cross_validation=20, percent_test_set=20):
        """
        Splits the dataset into 60% training, 20% cross-validation, 20% test set
        :return:
        """
        #TODO



    def BatchGradientDescent(self, X, y, batch_size, max_epochs, stop_when_error_less_than = 0.001):
        numSplits = X.shape[0] / batch_size
        X = np.array_split(X, numSplits, axis=0)
        y = np.array_split(y, numSplits, axis=0)

        plotData = []
        for i in range(1, max_epochs + 1):
            index = np.random.randint(0, len(X))
            self.nn.backprop(X[index], y[index], learningRate=0.03)
            error = self.nn.costFunction(y[index], self.nn.feedforward(X[index])[-1][-1])
            print("{} : Error {}".format(i, error))
            plotData.append(error)
            if(error < stop_when_error_less_than):
                break

        #plt.plot(plotData)

    def StochasticGradientDescent(self, X, y, max_epochs, stop_when_error_less_than = 0.001):
        self.BatchGradientDescent(X, y, X.shape[0], max_epochs, stop_when_error_less_than)

    def normalize(self, a):
        maxValues = np.amax(a, axis=0)
        return a / maxValues