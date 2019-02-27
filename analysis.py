import data
import numpy as np
import matplotlib.pyplot as plt

files, y_train = data.load_data()
y_augmented = np.concatenate((y_train, -y_train))

plt.hist(y_augmented, bins=[-1,-0.7,-0.5,-0.3,-0.1,0.1,0.3,0.5,0.7,1])
plt.show()