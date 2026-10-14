import numpy as np
import nnfs
#import rubiks as cub
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data
nnfs.init()
X, y = spiral_data(1000, 7)
class layerDense:
    def __init__(self, n_inputs, n_neurons) :
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros ((1, n_neurons))
    def forward(self, inputs) :
        self.output = np.dot (inputs, self.weights) + self.biases

class activationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
class activationSoftmax:
    def forward(self, logits):
        z = logits - logits.max(axis = 1, keepdims = True)
        e = np.exp(z)
        self.output = e / e.sum(axis = 1, keepdims = True)
class Loss:
    def calculate(self, output, y):
        sampleLosses = self.forward(output, y)
        dataLoss = np.mean(sampleLosses)
        return dataLoss
class lossCategoricalCrossEntropy(Loss):
    def forward(self, yPred, yTrue):
        samples = len(yPred)
        yPredClipped = np.clip(yPred, 1e-7, 1 - 1e-7)
        if len(yTrue.shape) == 1:
            correctConfidence = yPredClipped[range(samples), yTrue]
        elif len(yTrue.shape) == 2:
            correctConfidence = np.sum(yPredClipped*yTrue, axis = 1)
        negLoLikelyhood = -np.log(correctConfidence)
        return negLoLikelyhood

layer1 = layerDense(2, 3000)
activation1 = activationReLU()
layer1.forward(X)
activation1.forward(layer1.output)
layer2 = layerDense(3000, 3000)
activation2 = activationReLU()
layer2.forward(activation1.output)
activation2.forward(layer2.output)
layer3 = layerDense(3000, 30)
softmax = activationSoftmax()
layer3.forward(activation2.output)
softmax.forward(layer3.output)
#print(softmax.output)
#plt.plot(layer1.output, activation1.output, layer2.output, activation2.output)
lossFunction = lossCategoricalCrossEntropy()
loss = lossFunction.calculate(softmax.output, y)
predictions = np.argmax(softmax.output, axis = 1)
accuracy = np.mean(predictions == y) * 100 #y as class targets
print(softmax.output)
print("loss:", loss)
print("accuracy:", accuracy)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg", s=10)
plt.show()