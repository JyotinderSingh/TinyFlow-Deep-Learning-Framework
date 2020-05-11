import numpy as np
from Layers import Layer_Dense
import Activations
import Loss
import Model
import Optimizers


class Network:
    def __init__(self, inputFeatures):
        self.layers = []
        self.inputFeatures = inputFeatures
    
    def forwardPass(self):
        pass

    def backwardPass(self):
        pass

    def train(self, epochs):
        pass

    def addDenseLayer(self, neurons):
        if len(self.layers) == 0:
            denseX = Layer_Dense(self.inputFeatures, neurons)
            self.layers.append(denseX)
        else:
            denseX = Layer_Dense(
                self.layers[len(self.layers) - 1].weights.shape[1], neurons)
            self.layers.append(denseX)

    def getSummary(self):
        summary = ""
        for i in range(len(self.layers)):
            summary += f"Layer {str(i)} <" + self.layers[i].__str__() + ">\n"
        summary = summary.strip()
        return summary
