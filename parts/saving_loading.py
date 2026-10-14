import os
def Model(path="cubeModel.npz"):
    np.savez(path, w1=dense1.weights, b1=dense1.biases,
             w2=dense2.weights, b2=dense2.biases)

def loadModel(path="cubeModel.npz"):
    data = np.load(path)
    dense1.weights, dense1.biases, = data["w1"], data["b1"]
    dense2.weights, dense2.biases, = data["w2"], data["b2"]
    dense3.weights, dense3.biases = data["w3"], data["b3"]


if os.path.exists("cubeModel.npz"):
    loadModel()