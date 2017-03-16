import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork(object):
    def __init__(self, sizes, activationFunction):
        # sizes = [input_size, middle_size, output_size]

        self.num_layers = len(sizes)
        self.num_hidden_layers = self.num_layers - 2
        self.n = sizes[0]
        self.sizes = sizes
        self.activationFunction = activationFunction
        #self.costFunction = costFunction

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
        deltas[-1] = np.multiply(-(y - yHat), self.activationFunction(zs[-1], derivative=True))

        # all layers delta except last one
        for layer in range(self.num_hidden_layers - 1, -1, -1):
            deltas[layer] = np.multiply(np.dot(deltas[layer + 1], self.weights[layer + 1].T),
                                        self.activationFunction(zs[layer + 1], derivative=True))

        # weights updates (deltas * activation of previous layer)
        for layer in range(self.num_hidden_layers, -1, -1):
            dJW = np.dot(activations[layer].transpose(), deltas[layer])
            self.weights[layer] -= learningRate * dJW


def squaredError(y, yHat, derivative=False):
    #if derivative:
    #    return -(y - yHat)
    #else:
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
    def __init__(self):
        np.random.seed(1)  # to be reproductible
        self.nn = NeuralNetwork([1, 30, 30, 20, 1], tanh)

        sineRange = 2*np.pi
        sineMin = 0
        X, y = SampleDataset().getSinDataset(min=sineMin, sineRange=sineRange)
        self.BatchGradientDescent(X, y, batch_size=5, max_epochs=10000, stop_when_error_less_than=0.00001)

        plt.scatter((X * sineRange + sineMin) / 0.0174533, y)

        plt.scatter((X * sineRange + sineMin) / 0.0174533, self.nn.feedforward(X)[-1][-1], c='red')


    def SplitDataset(self):
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
            error = squaredError(y[index], self.nn.feedforward(X[index])[-1][-1])
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

class SampleDataset(object):
    def getSinDataset(self, min = 0, sineRange = np.pi):
        X = np.random.rand(1000, 1)
        y = np.sin(X * sineRange + min)
        return X, y

    def getXORDataset(self):
        X = np.array(([1, 1], [0, 1], [1, 0], [0, 0]), dtype=float)
        y = np.array(([0], [1], [1], [0]), dtype=float)

        return X, y

trainer = Trainer()
