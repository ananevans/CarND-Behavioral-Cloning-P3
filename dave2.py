import data
import numpy as np
            
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Cropping2D, BatchNormalization

import tensorflow as tf
tf.Session(config=tf.ConfigProto(log_device_placement=True))

# used implementation from https://github.com/TaavishThaman/LeNet-5-with-Keras/blob/master/lenet_5.py
def dave2():
    model = Sequential()
    model.add(BatchNormalization(input_shape=(160,320,3)))
    # cropping
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
    
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
    
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
    
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
    
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    
    # flatten
    model.add(Flatten())
    
    # dense 1164
    model.add(Dense(1164))
        
    # dense 100
    model.add(Dense(100))
    
    # dense 50
    model.add(Dense(50))
    
    # dense 10
    model.add(Dense(10))
    
    # dense 1
    model.add(Dense(1))
    
    model.compile(loss="mse", optimizer='adam')
    
    return model

import cv2 as cv

files, y_train = data.load_data()
X_train = []
for filename in files:
    X_train.append(cv.imread(filename))
X_train = np.array(X_train)
X_flipped, y_flipped = data.flip_images(X_train, y_train)

X_augmented = np.concatenate((X_train, X_flipped))
y_augmented = np.concatenate((y_train, y_flipped))

model = dave2()
model.fit(X_augmented, y_augmented, validation_split=0.2, shuffle=True, epochs=5)

model.save('dave2.h5')
