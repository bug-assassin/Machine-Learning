# http://gekkoquant.com/2016/03/13/evolving-neural-networks-through-augmenting-topologies-part-1-of-4/
import operator
import random

import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random as rd

import time

random.seed(10)
mutationRate = 1  # todo have it in a range of steps decreasing as time goes on
newNodeRandomRange = 2

ConnectionWeightMutateChance = 0.25
PerturbChance = 0.90
CrossoverChance = 0.75
LinkMutationChance = 2.0
NodeMutationChance = 0.50
BiasMutationChance = 0.40
StepSize = 0.1
DisableMutationChance = 0.4
EnableMutationChance = 0.2


class Node(object):
    def __init__(self, newGUID):
        self.GUID = newGUID
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
# Genes = links and node
class Genome(object):
    def __init__(self, inputSize, outputSize):
        self.nodes = []
        self.connectionNodes = []
        self.innovationNumber = 0
        self.inputNodes = []
        self.outputNodes = []
        self.curGUID = 64
        self.sizes = [inputSize, outputSize]

        for i in range(inputSize):
            newNode = Node(self.getNewNodeGUID())
            self.inputNodes.append(newNode)
            self.nodes.append(newNode)

        for i in range(outputSize):
            newNode = Node(self.getNewNodeGUID())
            self.outputNodes.append(newNode)
            self.nodes.append(newNode)

        # Create random connections between input and output
        for outNode in self.outputNodes:
            nodeIndex = random.randrange(len(self.inputNodes))
            selectedNode = self.inputNodes[nodeIndex]
            self.createLink(selectedNode, outNode)

    def getNewNodeGUID(self):
        self.curGUID += 1
        return chr(self.curGUID)

    def activate(self, X):
        if len(X) != len(self.inputNodes):
            raise Exception("Input does not match neural network number of input nodes")

        # Reset computed status at each activation
        for i in self.nodes:
            i.computed = False

        for i in range(len(X)):
            self.inputNodes[i].value = X[i]
            self.inputNodes[i].computed = True

        for node in self.inputNodes:
            node.propagate()

        returnValues = []
        for node in self.outputNodes:
            returnValues.append(node.value)

        return returnValues

    def mutate(self):
        if randomChancePassed(ConnectionWeightMutateChance):
            randomIndex = random.randrange(len(self.connectionNodes))
            self.pointMutate(self.connectionNodes[randomIndex])

        if randomChancePassed(LinkMutationChance):
            self.linkMutate()

        if randomChancePassed(NodeMutationChance):
            randomIndex = random.randrange(len(self.connectionNodes))
            self.nodeMutate(self.connectionNodes[randomIndex])

        if randomChancePassed(EnableMutationChance):
            self.enableDisableMutate()

    #
    # Mutations
    #

    # Randomly Change weight of the connection
    def pointMutate(self, conNode: ConnectionNode):
        conNode.weight += random.uniform(-mutationRate, mutationRate)

    # Creates new random link somewhere between unconnected nodes
    def linkMutate(self):
        # TODO connect only unconnected nodes
        inNode = self.nodes[random.randrange(len(self.nodes))]
        outNode = self.nodes[random.randrange(len(self.nodes))]

        # TODO make it o(1)
        if outNode in self.inputNodes:
            return

        # TODO make it o(1)
        for edge in self.connectionNodes:
            if edge.inNode == inNode and edge.outNode == outNode:
                return

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

        newNode = Node(self.getNewNodeGUID())
        self.createLink(oldConnection.inNode, newNode, 1)
        self.createLink(newNode, oldConnection.outNode, oldConnection.weight)

        self.nodes.append(newNode)

    # Toggles the enabled status of a random connection
    def enableDisableMutate(self):
        conNode = self.connectionNodes[random.randrange(len(self.connectionNodes))]
        conNode.enabled = not conNode.enabled

        #
        # End Mutations
        #

def randomChancePassed(self, percent):
    return rd.random() < percent

def display(network):
    graph = nx.DiGraph()

    for node in network.nodes:
        color = '#d3d3d3'
        if node in network.inputNodes:
            color = 'g'
        elif node in network.outputNodes:
            color = 'r'

        graph.add_node(node.GUID, node_color=color)

    for edge in network.connectionNodes:  # TODO ENABLED
        graph.add_edge(edge.inNode.GUID, edge.outNode.GUID, weight=edge.weight)

    colors = []
    for node in graph.nodes(data=True):
        colors.append(node[1]["node_color"])

    edgeColors = []
    for edge in graph.edges(data=True):
        w = edge[2]['weight']

        edgeColors.append(w)
        #if w < 0:
        #    edgeColors.append('r')
        #else:
        #    edgeColors.append('g')

    nx.draw_networkx(graph, node_color=colors, edge_color=edgeColors, width=3.0, edge_cmap=plt.cm.Blues)


neuralNetwork = Genome(inputSize=2, outputSize=1)
debug = False
for i in range(10):
    neuralNetwork.mutate()
    if debug:
        plt.clf()
        display(neuralNetwork)
        plt.pause(4)

# Genome Crossover
# genomes genes lined up using innovation number
# if innovation number not present in other genes
# It is a disjoint gene
# add it to the child


dataX = [[0, 0], [0, 1], [1, 0], [1, 1]]
dataY = [[0], [1], [1], [1]]


def fitnessFunction(y, expectedY):
    sumFitness = 1  # So we don't divide by 0 and the fitness will be between 0 and 1
    for index, val in enumerate(y):
        sumFitness += 1 / 2 * sum(map(operator.sub, val, expectedY[index])) ** 2

    return 1 / sumFitness


def calculateGenomeOutput(genome: Genome, X):
    result = []
    for input in X:
        result.append(genome.activate(input))

    return result


print(fitnessFunction(calculateGenomeOutput(neuralNetwork, dataX), dataY))

#display(neuralNetwork)
#plt.show()

def crossover(g1, g2):
    g1Fitness = fitnessFunction(calculateGenomeOutput(g1, dataX), dataY)
    g2Fitness = fitnessFunction(calculateGenomeOutput(g2, dataX), dataY)

    #Make sure g1 is highest fitness
    if g2Fitness > g1Fitness:
        temp = g1
        g1 = g2
        g2 = temp

    child = Genome(g1.sizes[0], g1.sizes[1])
    # Genes that match up are randomly selected from g1 or g2
    for gene1, gene2 in zip(g1.connectionNodes, g2.connectionNodes):
        if randomChancePassed(0.5):
            child.connectionNodes.append(gene1)
        else:
            child.connectionNodes.append(gene2)


    #Excess genes are taken from highest fitness parent (G1)





# Genome pool
# measuring similarity from weighted_sum of # disjoint and excess genes and difference in weights between matching genes
# if sum < threshold, same species

# Create genome pool
# Calculate fitness
# Assign genome to species
# Cull genomes
# Breed each species
# Repeat
