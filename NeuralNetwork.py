import numpy as np


class NeuralNetwork(object):
    def __init__(self, sizes, activationFunction):
        # sizes = [input_size, middle_size, output_size]

        self.num_layers = len(sizes)
        self.num_hidden_layers = self.num_layers - 2
        self.n = sizes[0]
        self.sizes = sizes
        self.activationFunction = activationFunction

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

    def costFunction(self, y, yHat):
        J = 0.5 * sum((y - yHat) ** 2)
        return J


def sigmoid(z, derivative=False):
    s = 1 / (1 + np.exp(-z))

    if derivative:
        return s * (1 - s)
    return s

def tanh(z, derivative=False): # doesn't work properly?
    s = 2 / (1 + np.exp(-2*z)) - 1

    if derivative:
        return 1 - s**2
    return s

class Trainer(object):
    np.random.seed(1)
    nn = NeuralNetwork([2, 3, 1], tanh)

    X = np.array(([1, 1], [0, 1], [1, 0], [0, 0]), dtype=float)
    y = np.array(([0], [1], [1], [0]), dtype=float)

    for _ in range(1, 200):
        nn.backprop(X, y, learningRate=0.2)
        print(nn.costFunction(y, nn.feedforward(X)[-1][-1]))

# assert nn.feedforward(np.array([1, 1]))[-1][-1] > 0.5
# assert nn.feedforward(np.array([1, 0]))[-1][-1] < 0.5
# assert nn.feedforward(np.array([0, 1]))[-1][-1] < 0.5
# assert nn.feedforward(np.array([0, 0]))[-1][-1] < 0.5
