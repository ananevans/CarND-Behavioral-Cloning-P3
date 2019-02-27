import data
import numpy as np
import matplotlib.pyplot as plt

samples = data.load_data()
y_train = samples[:,2]
print(y_train.shape)
y_train = y_train.astype(float)
print(y_train)

plt.hist(y_train, bins=[-1,-0.7,-0.5,-0.3,-0.1,0.1,0.3,0.5,0.7,1])
plt.show()