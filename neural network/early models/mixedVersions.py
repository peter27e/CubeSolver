import random
import numpy as np
import matplotlib.pyplot as plt


class layerDense:
    def __init__(self, nin, nout) :
        self.weights = 0.01 * np.random.randn(nin, nout)
        self.biases = np.zeros((1, nout))

    def forward(self, inputs) :
        self.inputs = inputs
        self.output = inputs @ self.weights + self.biases

    def backward(self, dvalues):
        self.dweights = self.inputs.T @ dvalues
        self.dbiases = np.sum(dvalues, axis = 0, keepdims = True)
        self.dinputs = dvalues @ self.weights.T

class activationReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class softmaxCrossEntropy:
    def __init__(self):
        self.loss = lossCategoricalCrossEntropy()

    def forward(self, inputs, ytrue):
        exp = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        self.output = exp / np.sum(exp, axis = 1, keepdims = True)
        return self.loss.calculate(self.output, ytrue)

    def backward(self, ytrue):
        samples = len(self.ouput)
        if len(ytrue.shape) == 2:
            ytrue = np.argmax(ytrue, axis = 1)
        self.dinputs = self.outputs.copy()
        self.dinputs[range(samples), ytrue] -= 1
        self.dinputs /= samples



#updater suggested by claude:
class optimizerSGD:
    def __init__(self, lr=0.1):
        self.lr = lr

    def update(self, layer):
        layer.weights -= self.lr * layer.dweights
        layer.biases -= self.lr * layer.dbiases








class Loss:
    def calculate(self, output, ys):
        sampleLosses = self.forward(output, ys)
        dataLoss = np.mean(sampleLosses)
        return dataLoss
    
class lossCategoricalCrossEntropy(Loss):
    def forward(self, ypred, yTrue):
        samples = len(ypred)
        yPredClipped = np.clip(ypred, 1e-7, 1 - 1e-7)
        if len(yTrue.shape) == 1:
            correctConfidence = yPredClipped[range(samples), yTrue]
        elif len(yTrue.shape) == 2:
            correctConfidence = np.sum(yPredClipped*yTrue, axis = 1)
        negLoLikelyhood = -np.log(correctConfidence)
        return negLoLikelyhood





dense1 = layerDense(324, 128)
relu1 = activationReLU()
dense2 = layerDense(128, 12)
lossAct = softmaxCrossEntropy()
optimizer = optimizerSGD(lr=0.1)


for round in range(tobedecided):
    #forward
    dense1.forward(X)
    relu1.forward(dense1.output)
    dense2.forward(relu1.output)
    loss = lossAct.forward(dense2.output, y)


    #backprop
    lossAct.backward(y)
    dense2.backward(lossAct.dinputs)
    relu1.backward(dense2.dinputs)
    dense1.backward(relu1.dinputs)


    #update
    optimizer.update(dense1)
    optimizer.update(dense2)

    print(round, loss)