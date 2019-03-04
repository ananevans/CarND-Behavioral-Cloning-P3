import glob
import numpy as np
import csv
import cv2 as cv
import dave2
import data
            
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D

from sklearn.model_selection import train_test_split

import tensorflow as tf
tf.Session(config=tf.ConfigProto(log_device_placement=True))


samples = data.load_data(True, True)
images, angles = data.load_images(samples)

X_train, X_val, y_train, y_val = train_test_split(images, angles, test_size=0.1)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)

model = dave2.dave2()
model.fit(X_train, y_train, validation_data = (X_val, y_val), nb_epoch=10, shuffle=True) 

model.save('model.h5')

