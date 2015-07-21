import pdb

import mnist_loader
import network2
from network2 import load
from plotter import plot_mnist_digit
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
#for data in training_data:
#	plot_mnist_digit(data[0])
#	#pdb.set_trace()
print 'data loaded'
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
net.save('network.json')
net.SGD(training_data, 1, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)
net.save('network.json')