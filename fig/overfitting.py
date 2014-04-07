"""
overfitting
~~~~~~~~~~~

Plot graphs to illustrate the problem of overfitting.
"""

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np

epochs = np.arange(40, 100, 1)

# The cost and accuracy data is generated using network2.py
cost = [95.64940534, 92.77003957, 91.19426513, 90.05272327,
        86.65961805, 83.72109886, 82.23502042, 79.32343131,
        77.48936465, 76.45793827, 74.29915603, 70.78666687,
        69.14013225, 67.60759576, 65.92640537, 64.37140944,
        62.64708059, 60.8062498, 59.73079664, 58.39325199,
        57.90915805, 55.93226466, 54.80526382, 53.84637318,
        52.58869068, 51.52430656, 50.64235016, 49.68848763,
        48.71468605, 47.8280628, 47.00485029, 46.28861009,
        45.81707749, 44.70239253, 43.85854394, 43.67765765,
        42.34793966, 41.7182567, 41.07244276, 40.71782626,
        40.07246729, 39.17786677, 38.78476657, 38.04577764,
        37.51645259, 36.97442833, 36.44307367, 35.91951676,
        35.48611018, 34.96091884, 34.59400404, 33.96090448,
        33.56307762, 33.1954933, 32.72731425, 32.22339415,
        31.79197639, 31.72643205, 31.03274438, 30.71485425]

accuracy = [82.92, 82.78, 82.93, 82.94, 82.92,
            82.98, 83.07, 83.05, 82.92, 83.02,
            83.37, 83.24, 82.94, 83.11, 83.07,
            83.34, 83.34, 83.27, 83.25, 83.31,
            83.44, 83.33, 83.41, 83.32, 83.30,
            83.30, 83.34, 83.55, 83.53, 83.51,
            83.35, 83.37, 83.41, 83.54, 83.45,
            83.50, 83.44, 83.50, 83.44, 83.48,
            83.37, 83.49, 83.41, 83.47, 83.42,
            83.46, 83.51, 83.54, 83.55, 83.38,
            83.48, 83.49, 83.57, 83.56, 83.39,
            83.32, 83.41, 83.54, 83.32, 83.41]
            

#Plot the cost data
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(epochs, cost)
ax.set_ylim([0, 100])
ax.set_xlim([40, 100])
ax.grid(True)
ax.set_xlabel('Epoch')
ax.set_title('Cost')
plt.show()

# Plot the accuracy data
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(epochs, accuracy)
ax.set_ylim([82, 84])
ax.set_xlim([40, 100])
ax.grid(True)
ax.set_xlabel('Epoch')
ax.set_title('Accuracy (%)')
plt.show()