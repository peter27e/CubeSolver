
import numpy as np
import random

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
    steps = random.randint(20, 30)
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

def checksolved():
    return np.array_equal(faces, solved)

def checkloop(maxMoves=100):
    moves = 0
    while not checksolved():
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



