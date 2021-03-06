import data
import train
import numpy as np
            
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Dropout, Cropping2D
from keras import regularizers

import tensorflow as tf
tf.Session(config=tf.ConfigProto(log_device_placement=True))

track1 = True
side_cameras = True

def dave2():
    model = Sequential()
    # normalization
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    # cropping
    model.add(Cropping2D(cropping=((60,20), (0,0))))

    #model.add(BatchNormalization(input_shape=(160,320,3)))
    
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu", padding="valid", kernel_regularizer=regularizers.l2(0.01)))
    
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu", padding="valid", kernel_regularizer=regularizers.l2(0.01)))
    
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu", padding="valid", kernel_regularizer=regularizers.l2(0.01)))
    
    model.add(Conv2D(64, (3, 3), activation="relu", padding="valid", kernel_regularizer=regularizers.l2(0.01)))
    
    model.add(Conv2D(64, (3, 3), activation="relu", padding="valid", kernel_regularizer=regularizers.l2(0.01)))
    
    # flatten
    model.add(Flatten())
        
    # dense 100
    model.add(Dense(100))
    model.add(Dropout(0.2))
    
    # dense 50
    model.add(Dense(50))
    model.add(Dropout(0.2))
    
    # dense 10
    model.add(Dense(10))
    model.add(Dropout(0.2))
    
    # dense 1
    model.add(Dense(1))
    
    model.compile(loss="mse", optimizer='adam', metrics=['accuracy'])
    
    print(model.summary())
    
    return model

#train.train( dave2(), 'dave2-dropout', True, False, True, 10)
#train.train( dave2(), 'dave2-dropout', True, True, True, 10)
#train.train( dave2(), 'dave2-dropout', False, False, True, 40)
#train.train( dave2(), 'dave2-dropout', False, True, True, 10)
