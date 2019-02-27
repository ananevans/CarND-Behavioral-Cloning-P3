import data
import numpy as np
            
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Cropping2D, BatchNormalization

import tensorflow as tf
from builtins import True
tf.Session(config=tf.ConfigProto(log_device_placement=True))

track1 = True
side_cameras = False

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

samples = data.load_data(track1, side_cameras)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

train_generator = data.generator(train_samples, batch_size=32)
validation_generator = data.generator(validation_samples, batch_size=32)

model = dave2()
#model.fit(X_augmented, y_augmented, validation_split=0.2, shuffle=True, epochs=5)
model.fit_generator(train_generator, samples_per_epoch = len(train_samples), 
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples), nb_epoch=5)

if track1:
    if side_cameras:
        name = 'dave2_track1_sides'
    else:
        name = 'dave2_track1'
else:
    if side_cameras:
        name = 'dave2_all_sides'
    else:
        name = 'dave2_all'

model.save(name)
