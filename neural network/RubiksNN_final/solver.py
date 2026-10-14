import os
import random
import numpy as np
from finalNN import (faces, movesDict, reset, encode, cksolved, printCube, loadModel,
                     dense1, relu1, dense2, dense3, relu2)

if os.path.exists("cubeModel.npz"):
    loadModel("cubeModel.npz")

movenames = ["R", "U", "F", "B", "L", "D", "R'", "U'", "F'", "B'", "L'", "D'"]
IDX = {m: i for i, m in enumerate(movenames)}

def inverse(m):
    if m.endswith("'"): return m[0]
    return m + "'"

INV = np.array([IDX[inverse(m)] for m in movenames])

def legalmask(prev):
    mask = np.ones(len(movenames), dtype=bool)
    if prev is not None:
        mask[INV[prev]] = False
    return mask

def pickmove(logits, prev, greedy=True):
    logits = np.where(legalmask(prev), logits, -np.inf)
    if greedy:
        return int(np.argmax(logits))
    p = np.exp(logits - logits.max())
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))

def scramble2(n):# shorter scramble func, non random moves count
    reset()
    moves = []
    prev = None
    for i in range(n):
        m = int(np.random.choice(np.flatnonzero(legalmask(prev))))
        movesDict[m]()
        moves.append(m)
        prev = m
    return moves

def predictlogits(X):
    X = np.asarray(X, dtype=float).reshape(1, -1)
    dense1.forward(X)
    relu1.forward(dense1.output)
    dense2.forward(relu1.output)
    relu2.forward(dense2.output)
    dense3.forward(relu2.output)
    out = dense3.output
    assert out.shape[1] == 12, f"expected 12 outputs, got {out.shape[1]}"
    return out[0]

def ckloop(mM=30, verbose=True):
    mvs = []
    prev = None
    while not cksolved():
        if len(mvs) >= mM:
            if verbose: print("gave up")
            return False, mvs
        m = pickmove(predictlogits(encode(faces)), prev)
        movesDict[m]()
        mvs.append(m)
        prev = m
    if verbose: print(f"solved in {len(mvs)} moves!!!!")
    return True, mvs

# cube steps displayed
scr = scramble2(6)
print("scramble:", " ".join(movenames[m] for m in scr))
printCube(faces)
ok, sol = ckloop()
print("solution", " ".join(movenames[m] for m in sol))

for n in range(1, 11):
    wins = 0
    for _ in range(200):
        scramble2(n)
        wins += ckloop(verbose=False)[0]
    print(f"depth {n}: {wins/200:.1%}")
