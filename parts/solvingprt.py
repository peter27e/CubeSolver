def scramble2(n): #shorter scramble func, non random moves count
    reset()
    moves = []
    for i in range(n):
        m = random.randint(0, 11)
        movesDict[m]()
        moves.append(m)
    return moves

def predictmove(x):
    dense1.forward(x)
    relu1.forward(dense1.output)
    dense2.forward(relu1.output)
    relu2.forward(dense2.output)
    dense3.forward(relu2.output)
    out = dense3.output
    assert out.shape[1] == 12, f"expected 12 outputs, got {out.shape[1]}"
    return int(np.argmax(out, axis=1)[0])

def ckloop(mM=30, verbose=True):
    moves = []
    while not cksolved():
        if len(moves) >= mM:
            if verbose: print("gave up")
            return False, moves
        m = predictmove()
        movesDict[m]()
        moves.append(m)
    if verbose: print(f"solved in {len(moves)} moves!!!!")
    return True, moves