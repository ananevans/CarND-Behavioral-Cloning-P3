import data
import train
import numpy as np
            
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Cropping2D, BatchNormalization

import tensorflow as tf
tf.Session(config=tf.ConfigProto(log_device_placement=True))

# used implementation from https://github.com/TaavishThaman/LeNet-5-with-Keras/blob/master/lenet_5.py
def lenet():
    model = Sequential()
    model.add(BatchNormalization(input_shape=(160,320,3)))
    # cropping
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
    #Layer 1
    #Conv Layer 1
    model.add(Convolution2D(filters = 6, 
                     kernel_size = 5, 
                     strides = 1, 
                     activation = 'relu', 
                     input_shape = (90,320,3)))
    #Pooling layer 1
    model.add(MaxPooling2D(pool_size = 2, strides = 2))
    #Layer 2
    #Conv Layer 2
    model.add(Convolution2D(filters = 16, 
                     kernel_size = 5,
                     strides = 1,
                     activation = 'relu',
                     input_shape = (14,14,6)))
    #Pooling Layer 2
    model.add(MaxPooling2D(pool_size = 2, strides = 2))
    #Flatten
    model.add(Flatten())
    #Layer 3
    #Fully connected layer 1
    model.add(Dense(units = 120, activation = 'relu'))
    #Layer 4
    #Fully connected layer 2
    model.add(Dense(units = 84, activation = 'relu'))
    #Layer 5
    #Output Layer
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

train.train(lenet(), 'lenet', True, True, False, 3)
train.train(lenet(), 'lenet', True, False, False, 3)
train.train(lenet(), 'lenet', False, True, False, 3)
train.train(lenet(), 'lenet', False, False, False, 3)
