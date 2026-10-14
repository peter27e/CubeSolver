import numpy as np
def stats_PredAccu():
    softmaxOutputs = np.array([[0.1, 0.444, 0.2],
                            [0.8, 0.213, 0.6],
                            [0.5, 0.09, 0.112]])
    classTargets = [0, 0, 1]
    predictions = np.argmax(softmaxOutputs, axis = 1)
    accuracy = np.mean(predictions == classTargets)
    return accuracy
