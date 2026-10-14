import os
import random
import numpy as np


#rubix sim
faces = np.array([[[f]*3 for i in range(3)]for f in range(6)]) # 6x3x3
solved = faces.copy()
whiteFace = faces[0]
yellowFace = faces[1]
blueFace = faces[2]
redFace = faces[3]
orangeFace = faces[4]
greenFace = faces[5]

def R():
    ogWFace = whiteFace.copy()
    whiteFace[:, -1] = greenFace[:, -1]
    greenFace[:, -1] = yellowFace[:, -1]
    yellowFace[:, -1] = blueFace[:, 0][::-1]
    blueFace[:, 0] = ogWFace[:, -1][::-1]
    redFace[:] = np.rot90(redFace, -1)
def U():
    ogOFace = orangeFace.copy()
    orangeFace[0] = greenFace[0]
    greenFace [0] = redFace[0]
    redFace[0] = blueFace[0]
    blueFace[0] = ogOFace[0]
    whiteFace[:] = np.rot90(whiteFace, -1)
def F():
    ogWFace = whiteFace.copy()
    ogYFace = yellowFace.copy()
    ogRFace = redFace.copy()
    whiteFace[2] = orangeFace[:, -1][::-1]
    redFace[:, 0] = ogWFace[2]
    yellowFace[0] = ogRFace[:, 0][::-1]
    orangeFace[:, -1] = ogYFace[0]
    greenFace[:] = np.rot90(greenFace, -1)
def B():
    ogWFace = whiteFace.copy()
    whiteFace[0] = redFace[:, -1]
    redFace[:, -1] = yellowFace[2][::-1]
    yellowFace[2] = orangeFace[:, 0]
    orangeFace[:, 0] = ogWFace[0][::-1]
    blueFace[:] = np.rot90(blueFace, -1)
def L():
    ogWFace = whiteFace.copy()
    whiteFace[:, 0] = blueFace[:, -1][::-1]
    blueFace[:, -1] = yellowFace[:, 0][::-1]
    yellowFace[:, 0] = greenFace[:, 0]
    greenFace[:, 0] = ogWFace[:, 0]
    orangeFace[:] = np.rot90(orangeFace, -1)
def D():
    ogBFace = blueFace.copy()
    blueFace[2] = redFace[2]
    redFace[2] = greenFace[2]
    greenFace[2] = orangeFace[2]
    orangeFace[2] = ogBFace[2]
    yellowFace[:] = np.rot90(yellowFace, -1)


#reversed moves
def Rr(): R(), R(), R()
def Ru(): U(), U(), U()
def Rf(): F(), F(), F()
def Rb(): B(), B(), B()
def Rl(): L(), L(), L()
def Rd(): D(), D(), D()

#for scramble
movesDict = {
    0 : R,
    1 : U,
    2 : F,
    3 : B,
    4 : L,
    5 : D,
    6 : Rr,
    7 : Ru,
    8 : Rf,
    9 : Rb,
    10 : Rl,
    11 : Rd,
} 

# more functions:


def reset():
    faces[:] = solved #i wonder what this does


def scramble():
    reset()
    states, moves = [], []
    steps = random.randint(10, 20)
    for i in range(steps):
        m = random.randint(0, 11)
        movesDict[m]()
        states.append(faces.copy())
        moves.append(m)
    return states, moves

def inverse(m):
    return (m+6)%12 #for sm

def encode(state):
    return np.eye(6)[state].flatten() #one hot 

def cksolved():
    return np.array_equal(faces, solved)

def ckloop(maxMoves=100):
    moves = 0
    while not cksolved():
        if moves >= maxMoves:
            print("gave up")
            return False
        moves += 1
    print("solved!!!!")
    return True


def printCube(faces):
    white, yellow, blue, red, orange, green = faces
    row = lambda r: " ".join(str(x) for x in r)
    gap = "   "
    pad = " " * 5 + gap
    for r in white:
        print(pad + row(r))
    print()
    for i in range(3):
        print(gap.join(row(f[i]) for f in (orange, green, red, blue)))
    print()
    for r in yellow:
        print(pad + row(r))



print(scramble())
printCube(faces)
#connecting vars
nScrambles = 4000
xlist, ylist = [], []
for _ in range(nScrambles):
    states, moves = scramble()
    for s, m in zip(states, moves):
        xlist.append(encode(s))
        ylist.append(inverse(m))

X = np.array(xlist)
y = np.array(ylist)
#print(X.shape, y.shape, y.dtype)
#print(y.min(), y.max())
#print(X.sum(axis=1)[:5])


#neural network structure
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
        samples = len(self.output)
        if len(ytrue.shape) == 2:
            ytrue = np.argmax(ytrue, axis = 1)
        self.dinputs = self.output.copy()
        self.dinputs[range(samples), ytrue] -= 1
        self.dinputs /= samples

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





#updater
class optimizerSGD:
    def __init__(self, lr=0.2):
        self.lr = lr

    def update(self, layer):
        layer.weights -= self.lr * layer.dweights
        layer.biases -= self.lr * layer.dbiases



def saveModel(path="cubeModel.npz"):
    np.savez(path, w1=dense1.weights, b1=dense1.biases,
                   w2=dense2.weights, b2=dense2.biases, 
                   w3=dense3.weights, b3=dense3.biases)

def loadModel(path="cubeModel.npz"):
    data = np.load(path)
    dense1.weights, dense1.biases = data["w1"], data["b1"]
    dense2.weights, dense2.biases = data["w2"], data["b2"]
    dense3.weights, dense3.biases = data["w3"], data["b3"]

dense1 = layerDense(324, 1028)
relu1 = activationReLU()
dense2 = layerDense(1028, 2056)
relu2 = activationReLU()
dense3 = layerDense(2056, 12)
lossAct = softmaxCrossEntropy()
optimizer = optimizerSGD(lr=0.2)
i=0
acc = 0

#trsining loop shape
if __name__ == "__main__":
    while acc<0.7:
        i+=1
        #forward
        dense1.forward(X)
        relu1.forward(dense1.output)
        dense2.forward(relu1.output)
        relu2.forward(dense2.output)
        dense3.forward(relu2.output)
        loss = lossAct.forward(dense3.output, y)


        #backprop
        lossAct.backward(y)
        dense3.backward(lossAct.dinputs)
        relu2.backward(dense3.dinputs)
        dense2.backward(relu2.dinputs)
        relu1.backward(dense2.dinputs)
        dense1.backward(relu1.dinputs)


        #update
        optimizer.update(dense1)
        optimizer.update(dense2)
        optimizer.update(dense3)

        acc = np.mean(np.argmax(lossAct.output, axis=1) == y)
        print(i, loss, acc)
    saveModel()

