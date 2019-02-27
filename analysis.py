import data
import numpy as np
import matplotlib.pyplot as plt

files, y_train = data.load_data()
plt.hist(y_train)
plt.show()