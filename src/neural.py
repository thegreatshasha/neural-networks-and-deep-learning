import numpy as np
import mnist_loader
import random

# Common helpers
def sigmoid(z):
        return 1.0/(1.0 + np.exp(-z))

sigmoid_vec = np.vectorize(sigmoid)


class Neural():
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = np.array([np.random.randn(x, 1) for x in sizes[1:]])
        self.weights = np.array([np.random.randn(y,x) for (x,y) in zip(sizes[:-1], sizes[1:])])
        print self.biases
        print self.weights

    def stochastic_gradient_descent(self, training_data, epoch_size, batch_size, eta, test_data):
        for x in xrange(0, epoch_size):
            print 'Running epoch {0} out of {1}'.format(x, epoch_size)
            random.shuffle()

    def update_batch(self, batch, eta):
        print batch, eta

    def evaluate(self, test_data):
        pass


    def forward(self, a):
        a = np.array(a)
        for b, w in zip(self.biases, self.weights):
            a = sigmoid_vec(np.dot(w, a) + b)
        return a
 

n = Neural([1,2,3])
print n.forward([[1]])

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import pdb; pdb.set_trace()
print 'done loading data'