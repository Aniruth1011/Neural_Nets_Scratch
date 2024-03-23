import numpy as np 
from layers import DenseLayer
from load_data import get_data

x_train, y_train, x_test, y_test = get_data()

    
class NeuralNetwork:
    def __init__(self):
        self.layer1 = DenseLayer(x_train.shape[1], 64)
        self.layer2 = DenseLayer(64, 32)
        self.layer3 = DenseLayer(32 , 16)
        self.layer4 = DenseLayer(16, 1)

    def forward(self, inputs):
        self.layer1.forward_prop(inputs)
        self.layer2.forward_prop(np.tanh(self.layer1.output))  
        self.layer3.forward_prop(np.tanh(self.layer2.output)) 
        self.layer4.forward_prop(np.tanh(self.layer3.output)) 

    def backward(self, y_true, learning_rate):
        doutput4 = 2 * (self.layer4.output - y_true) / y_true.shape[0]
        doutput3 = self.layer4.backward_prop(doutput4, learning_rate)
        doutput2 = self.layer3.backward_prop(doutput3, learning_rate)
        doutput1 = self.layer2.backward_prop(doutput2, learning_rate)

        self.layer1.backward_prop(doutput1, learning_rate)

    def train(self, X, y, learning_rate=0.01, epochs=200):
        for epoch in range(epochs):
            self.forward(X)
            loss = np.mean((self.layer4.output - y) ** 2) / 2 

            self.backward(y.reshape(-1, 1), learning_rate)
            print("Epoch: {}, Loss: {}".format(epoch+1, loss))


    def predict(self, X):
        self.forward(X)
        return self.layer4.output


model = NeuralNetwork()
model.train(x_train, y_train, learning_rate=0.01, epochs=200)
