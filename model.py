import glob
import numpy as np
import csv
import cv2 as cv
import dave2
            
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D

from sklearn.model_selection import train_test_split

import tensorflow as tf
tf.Session(config=tf.ConfigProto(log_device_placement=True))

data_dirs = glob.glob('/home/ans5k/work/CarND-Behavioral-Cloning-P3/data/*')

images = []
angles = []
data_home = '/home/ans5k/work/CarND-Behavioral-Cloning-P3/data/'
data_dirs = glob.glob(data_home + '*')
angle_adjustment = 0.1
for dir in data_dirs:
    with open(dir + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        image_path = dir + "/IMG/"
        for line in reader:
             center_image = cv.imread(image_path + line[0].split('/')[-1])
             center_image_rgb = cv.cvtColor(center_image, cv.COLOR_BGR2RGB)
             images.append(center_image_rgb)
             angles.append(float(line[3]))
             #flipped
             images.append(cv.flip(center_image_rgb, 1))
             angles.append(-float(line[3]))

             left_image = cv.imread(image_path + line[1].split('/')[-1])
             left_image_rgb = cv.cvtColor(center_image, cv.COLOR_BGR2RGB)
             images.append(left_image_rgb)
             angles.append(float(line[3])+angle_adjustment)
             #flipped
             images.append(cv.flip(left_image_rgb, 1))
             angles.append(-(float(line[3])+angle_adjustment))

             right_image = cv.imread(image_path + line[2].split('/')[-1])
             right_image_rgb = cv.cvtColor(center_image, cv.COLOR_BGR2RGB)
             images.append(right_image_rgb)
             angles.append(float(line[3])-angle_adjustment)
             #flipped
             images.append(cv.flip(right_image_rgb, 1))
             angles.append(-(float(line[3])-angle_adjustment))

X_train, X_val, y_train, y_val = train_test_split(images, angles, test_size=0.1)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)

model = dave2.dave2()
model.fit(X_train, y_train, validation_data = (X_val, y_val), nb_epoch=10, shuffle=True) 

model.save('model.h5')

