from NeuralNetwork import *
import numpy as np

class SampleDataset(object):
    def getSinDataset(self, min = 0, sineRange = np.pi):
        X = np.random.rand(1000, 1)
        y = np.sin(X * sineRange + min)
        return X, y

    def getXORDataset(self):
        X = np.array(([1, 1], [0, 1], [1, 0], [0, 0]), dtype=float)
        y = np.array(([0], [1], [1], [0]), dtype=float)

        return X, y

# Test sine
nn = NeuralNetwork([1,30,30,20,1], tanh, squaredError)
trainer = Trainer(nn)

sineRange = 2*np.pi
sineMin = 0
X, y = SampleDataset().getSinDataset(min=sineMin, sineRange=sineRange)
trainer.BatchGradientDescent(X, y, batch_size=5, max_epochs=10000, stop_when_error_less_than=0.00001)

plt.scatter((X * sineRange + sineMin) / 0.0174533, y)
plt.scatter((X * sineRange + sineMin) / 0.0174533, nn.feedforward(X)[-1][-1], c='red')