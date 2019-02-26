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
X_train = []
for filename in files:
    X_train.append(cv.imread(filename))
X_train = np.array(X_train)
X_flipped, y_flipped = data.flip_images(X_train, y_train)

X_augmented = np.concatenate(X_train, X_flipped)
y_augmented = np.concatenate(y_train, y_flipped)

model = basic_model()
model.fit(X_augmented, y_augmented, validation_split=0.2, shuffle=True, epochs=5)

model.save('basic_model.h5')



