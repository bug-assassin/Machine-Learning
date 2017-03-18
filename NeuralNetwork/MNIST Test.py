from mnist import MNIST
from NeuralNetwork import *
import matplotlib.pyplot as plt

mndata = MNIST("C:\\Users\\Work\\Documents\\GitHub\\NeuralNetworkBackprop\\mnist_dataset\\")
dataset_training = np.array(mndata.load_training())
#dataset_test = np.array(mndata.load_testing())

print(dataset_training.shape)
#print("\nTest\n")
#print(dataset_test.shape)

X = []
for i in range(len(dataset_training[0])):
    X.append(np.array(dataset_training[0, i]))
X = np.array(X) / 255

m = len(dataset_training[1])
y = np.zeros((m, 10), dtype=int)
for i in range(m):
    y[i][dataset_training[1][i]] = 1

nn = NeuralNetwork([784, 300, 10], sigmoid, squaredError)
trainer = Trainer(nn)
trainer.BatchGradientDescent(X, y, batch_size=20, max_epochs=500, stop_when_error_less_than=0)

images = []

# len(dataset_training[0])
for i in range(0, 10):
    images.append(np.reshape(dataset_training[0, i], (28, 28)).astype("uint8"))

#for i in range(0, 10):
#    plt.imshow(images[i], cmap=plt.cm.Greys)
#    print(dataset_training[1][i])
#    plt.pause(1)
def testImage(image):
    image = image / 255
    plt.imshow(image, cmap=plt.cm.Greys)
    # noinspection PyTypeChecker
    answer = nn.feedforward(np.reshape(image, 784))[-1][-1]
    print(answer, "\n")
    print("Answer: ", answer.argmax(axis=0))
