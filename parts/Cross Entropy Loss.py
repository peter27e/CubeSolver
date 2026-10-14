import math
import numpy as np
softmaxOutput = np.array([[0.2, 0.99, 0.2],
                          [0.3, 0.39, 0.9],
                          [0.02, 0.5, 0.98]])
classTargets = [0, 1, 1]
#loss = -(math.log(softmaxOutput[0]) * targetOutput[0]+
         #math.log(softmaxOutput[1]) * targetOutput[1]+
         #math.log(softmaxOutput[2]) * targetOutput[2])
yPredClipped = np.clip(yPred, 1e-7, 1 - 1e-7)
negLog = -np.log(softmaxOutput[
    range(len([0, 1, 2])), classTargets])
avarageLoss = np.mean(negLog)
print(negLog)
#loss = -(math.log(softmaxOutput[0]))
#print(loss)
