import os
import random
import numpy as np
from finalNN import (faces, movesDict, reset, encode, cksolved, printCube, loadModel, dense1, relu1, dense2, dense3, relu2)

if os.path.exists("cubeModel.npz"):
    loadModel("cubeModel.npz")
movenames = ["R", "U", "F", "B", "L", "D", "R'", "U'", "F'", "B'", "L'", "D'"]

def scramble2(n): #shorter scramble func, non random moves count
    reset()
    moves = []
    for i in range(n):
        m = random.randint(0, 11)
        movesDict[m]()
        moves.append(m)
    return moves

def predictmove(X):
    X = np.asarray(X, dtype=float).reshape(1, -1)
    dense1.forward(X)
    relu1.forward(dense1.output)
    dense2.forward(relu1.output)
    relu2.forward(dense2.output)
    dense3.forward(relu2.output)
    out = dense3.output
    assert out.shape[1] == 12, f"expected 12 outputs, got {out.shape[1]}"
    return int(np.argmax(out, axis=1)[0])

def ckloop(mM=30, verbose=True):
    mvs = []
    while not cksolved():
        if len(mvs) >= mM:
            if verbose: print("gave up")
            return False, mvs
        m = predictmove(encode(faces))
        movesDict[m]()
        mvs.append(m)
    if verbose: print(f"solved in {len(mvs)} moves!!!!")
    return True, mvs

#cube steps displayed
scr = scramble2(6)
print("scramble:", " ".join(movenames[m] for m in scr))
printCube(faces)
ok, sol = ckloop()
print(type(sol), sol[:3])
print("solution", " ".join(movenames[m] for m in sol))

for n in range(1, 11):
    wins = 0
    for _ in range(200):
        scramble2(n)
        wins+=ckloop(verbose=False)[0]
    print(f"depth {n}: {wins/200:.1%}")


