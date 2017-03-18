# http://gekkoquant.com/2016/03/13/evolving-neural-networks-through-augmenting-topologies-part-1-of-4/
import random
import numpy as np

random.seed(10)
mutationRate = 1  # todo have it in a range of steps decreasing as time goes on
newNodeRandomRange = 2

nodeGUID = 0

class Node(object):
    def __init__(self):
        global nodeGUID
        nodeGUID += 1
        self.GUID = nodeGUID

        self.inputCon = []
        self.outputCon = []
        self.computed = False
        self.value = 0

    def compute(self):
        if not self.computed:
            nodeSum = 0
            for connection in (x for x in self.inputCon if x.enabled):
                nodeSum += connection.inNode.value * connection.weight

            self.value = activationFunction(nodeSum)
            self.computed = True
            self.propagate()

    def propagate(self):
        for connection in (x for x in self.outputCon if x.enabled):
            connection.outNode.compute()


class ConnectionNode(object):
    def __init__(self, inNode: Node, outNode: Node, weight, innovationNumber: int, enabled=True):
        self.inNode = inNode
        self.outNode = outNode
        self.weight = weight
        self.enabled = enabled
        self.innovationNumber = innovationNumber


def activationFunction(z):
    return sigmoid(z)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Genome = neural network
# Genes = links and nodes
class Genome(object):
    def __init__(self, inputSize, outputSize):
        self.nodes = []
        self.connectionNodes = []
        self.innovationNumber = 0
        self.inputNodes = []
        self.outputNodes = []

        for i in range(inputSize):
            self.inputNodes.append(Node())

        for i in range(outputSize):
            self.outputNodes.append(Node())

        # Create random connections between input and output
        for outNode in self.outputNodes:
            nodeIndex = random.randrange(len(self.inputNodes))
            selectedNode = self.inputNodes[nodeIndex]
            self.createLink(selectedNode, outNode)

    # Randomly Change weight of the connection
    def pointMutate(self, conNode: ConnectionNode):
        conNode.weight += random.uniform(-mutationRate, mutationRate)

    # Creates new random link somewhere between unconnected nodes
    def linkMutate(self):
        # TODO connect only unconnected nodes
        inNode = self.nodes[random.randrange(len(self.nodes))]
        outNode = self.nodes[random.randrange(len(self.nodes))]
        self.createLink(inNode, outNode)

    def createLink(self, inNode, outNode, weight=None):
        if weight is None:
            weight = random.uniform(-newNodeRandomRange, newNodeRandomRange)

        self.innovationNumber += 1

        newConnection = ConnectionNode(inNode, outNode, weight, self.innovationNumber)
        inNode.outputCon.append(newConnection)
        outNode.inputCon.append(newConnection)

        self.connectionNodes.append(newConnection)

    # Creates new node between a, b
    def nodeMutate(self, oldConnection: ConnectionNode):
        oldConnection.enabled = False

        newNode = Node()
        self.createLink(oldConnection.inNode, newNode, 1)
        self.createLink(newNode, oldConnection.outNode, oldConnection.weight)

        self.nodes.append(newNode)

    # Toggles the enabled status of a random connection
    def enableDisableMutate(self):
        conNode = self.connectionNodes[random.randrange(len(self.connectionNodes))]
        conNode.enabled = not conNode.enabled

    def activate(self, X):
        if len(X) != len(self.inputNodes):
            raise Exception("Input does not match neural network number of input nodes")

        for i in range(len(X)):
            self.inputNodes[i].value = X[i]
            self.inputNodes[i].computed = True

        for node in self.inputNodes:
            node.propagate()

        returnValues = []
        for node in self.outputNodes:
            returnValues.append(node.value)

        return returnValues


neuralNetwork = Genome(inputSize=3, outputSize=3)
print(neuralNetwork.activate([1, 0, 1]))

# Genome Crossover
# genomes genes lined up using innovation number
# if innovation number not present in other genes
# It is a disjoint gene
# add it to the child


# Genome pool
# measuring similarity from weighted_sum of # disjoint and excess genes and difference in weights between matching genes
# if sum < threshold, same species

# Create genome pool
# Calculate fitness
# Assign genome to species
# Cull genomes
# Breed each species
# Repeat
