import numpy as np
import nnfs
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data
nnfs.init
np.random.seed(0)
X = [[2, -1.5, -3, 0.9],
   [1, 0.5, -1, -1.5],
   [1.1, 0.23, -0.5, 1.2]]

class layerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)*np.sqrt(2/n_inputs)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.outputs = np.dot(inputs, self.weights) + self.biases
class activationReLU():
    def forward(self, inputs):
        self.outputs = np.maximum(0, inputs)
class activationSoftMax:
    def forward(self, inputs):
        exp = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        self.outputs = exp/ np.sum(exp, axis = 1, keepdims = True)

layer1 = layerDense(3, 20); act1 = activationReLU()
layer2 = layerDense(20, 20); act2 = activationReLU()
layer3 = layerDense(20, 20); act3 = activationReLU()
layer4 = layerDense(20, 40); act4 = activationReLU()
layer5 = layerDense(40, 20); act5 = activationReLU()
layer6 = layerDense(20, 10); act6 = activationSoftMax()
layer1.forward(X); act1.forward(layer1.outputs)
layer2.forward(act1.outputs); act2.forward(layer2.outputs)
layer3.forward(act2.outputs); act3.forward(layer3.outputs)
layer4.forward(act3.outputs); act4.forward(layer4.outputs)
layer5.forward(act4.outputs); act5.forward(layer5.outputs)
layer6.forward(act5.outputs); act6.forward(layer6.outputs)
print(act1.outputs)
print(act2.outputs)
print(act3.outputs)
print(act4.outputs)
print(act5.outputs)
print(act6.outputs)
plt.plot(act1.outputs, act2.outputs, act3.outputs, act4.outputs, act5.outputs, act6.outputs)
plt.show()