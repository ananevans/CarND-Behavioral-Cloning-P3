import data
import train
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

train.train(basic_model(), 'sequential', True, True, False, 3)
train.train(basic_model(), 'sequential', True, False, False, 3)
train.train(basic_model(), 'sequential', False, True, False, 3)
train.train(basic_model(), 'sequential', False, False, False, 3)