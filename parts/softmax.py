import numpy as np
def softmax(x):
    def forward(self, inputs):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

class activationSoftMax:
    def forward(self, inputs):
        exp = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        self.outputs = exp/ np.sum(exp, axis = 1, keepdims = True)
