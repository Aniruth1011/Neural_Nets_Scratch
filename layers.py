import numpy as np 
from tqdm import tqdm 
class DenseLayer:

    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim)
        self.bias = np.zeros((1, output_dim))

    def forward_prop(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.bias

    def backward_prop(self, doutput, learning_rate):
        weights_d = np.dot(self.inputs.T, doutput)
        bias_d = np.sum(doutput, axis=0, keepdims=True)
        inputs_d =  np.dot(doutput, self.weights.T)

        self.weights -= learning_rate * weights_d
        self.bias -= learning_rate * bias_d 


        return inputs_d


