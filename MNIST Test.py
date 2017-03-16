from mnist import MNIST
from NeuralNetwork import *

mndata = MNIST("C:\\Users\\Work\\Documents\\GitHub\\NeuralNetworkBackprop\\mnist_dataset\\")
dataset_training = np.array(mndata.load_training())
#dataset_test = np.array(mndata.load_testing())

print(dataset_training.shape)
#print("\nTest\n")
#print(dataset_test.shape)

images = []

# len(dataset_training[0])
for i in range(0, 10):
    images.append(np.reshape(dataset_training[0, i], (28, 28)).astype("uint8"))

for i in range(0, 10):
    plt.imshow(images[i], cmap=plt.cm.Greys)
    print(dataset_training[1][i])
    plt.pause(1)