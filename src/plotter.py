import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plot_mnist_digit(image):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    image= np.array(image).reshape(28, 28)
    image = image.clip(min=0, max=1)
    print image
    ax.matshow(image, cmap = matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.show()