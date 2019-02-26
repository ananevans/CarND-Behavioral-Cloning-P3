import data
import numpy as np
            
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda

import tensorflow as tf
tf.Session(config=tf.ConfigProto(log_device_placement=True))

def basic_model():
    model = Sequential()
    # normalize and center the data
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
    
    model.add(Flatten(input_shape=(160,320,3)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

import cv2 as cv

files, y_train = data.load_data()
X_train = data.load_images(files)

model = basic_model()
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3)

model.save('basic_model.h5')
