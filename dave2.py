import data
import train
import numpy as np
            
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Cropping2D, BatchNormalization

import tensorflow as tf
tf.Session(config=tf.ConfigProto(log_device_placement=True))

track1 = True
side_cameras = True

def dave2():
    model = Sequential()
    # cropping
    model.add(Cropping2D(cropping=((55,20), (0,0)), input_shape=(160,320,3)))

    #model.add(BatchNormalization(input_shape=(160,320,3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu", padding="valid"))
    
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu", padding="valid"))
    
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu", padding="valid"))
    
    model.add(Conv2D(64, (3, 3), activation="relu", padding="valid"))
    
    model.add(Conv2D(64, (3, 3), activation="relu", padding="valid"))
    
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
    
    print(model.summary())
    
    return model

train.train( dave2(), 'dave2', True, True, False, 5)
train.train( dave2(), 'dave2', True, False, False, 5)
train.train( dave2(), 'dave2', False, True, False, 5)
train.train( dave2(), 'dave2', False, False, False, 5)
