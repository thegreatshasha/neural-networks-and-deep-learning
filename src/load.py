import pdb

import mnist_loader
from network import load
from plotter import plot_mnist_digit
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
#for data in training_data:
#	plot_mnist_digit(data[0])
#	#pdb.set_trace()
training_data_2 = [data for data in training_data if data[1][4] == 1]
pdb.set_trace()
print 'data loaded'
net = load('network.json')
net.generated_data  = training_data[0][0]
net.SGD(training_data_2, 30, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)