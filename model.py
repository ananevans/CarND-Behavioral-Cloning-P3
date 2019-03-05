import glob
import numpy as np
import csv
import cv2 as cv
import dave2_dropout
import train

            
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D

from sklearn.model_selection import train_test_split

train.train( dave2_dropout.dave2(), 'dave2-dropout', False, True, True, 10, 0.5)

