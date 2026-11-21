> [!WARNING]
> `cubeModel.npz` is outdated. It was trained on the old 3-layer network and doesn't work with the current
> 4-layer `finalNN.py`, so `solver.py` crashes with `KeyError: 'w4'` while it's there. No newly trained model yet,
> coming soon. Until then: delete `cubeModel.npz`, run `finalNN.py` from the `RubiksNN_final` folder to train your
> own (it saves a new `cubeModel.npz` after every cycle, so you can stop whenever), then run `solver.py`.
> Training is slow, expect 20-30 seconds per cycle.

Hello, i made a neural network ,using only numpy, random, and os, with the goal for it to solve a rubiks cube.

Maybe a bit exessive for a cube but its easier than writing 48 quintillion if (mostly elif) statements

Obviously i didnt come up with the structure for this, i used mainly Andrej Karpathy's video on micrograd, 
SentDex's videos on neural networks, and some claude to help me debug.

The folder "early models" is an unorganized list of different "models" i made for experimenting and following
along with youtube videos.

The folder "parts" was an attempt at keeping my code organized and being able to find any function quite fast, 
unfortunately its missing quite a bit, i wont bother to organize it.

If you want to have some fun with the NN and change the neuron count then make sure to update the saving and 
loading functions too, i lost some hours of training to that.


youtube links:

Andrej Karpathy: https://youtu.be/VMj-3S1tku0?si=eAyjxhaAKMo1HOE0

SentDex (link to ep 1, i watched 1-8): https://youtu.be/Wo5dMEP_BbI?si=hYWr-UA68BeiCCeR
