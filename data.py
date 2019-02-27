import csv
import numpy as np
import cv2 as cv
data_home = '/home/nora/work/CarND-Behavioral-Cloning-P3/data/'
#data_home = '/home/ans5k/work/CarND-Behavioral-Cloning-P3/data/'

def load_data():
    #data_dirs = ['data', 'track1', 'correction', 'track1_backwards', 'track2', 'track2_more', 'track1_curve', 'maxwell', 'maxwell2', 'maxwell_reverse', 'no_borders']
    data_dirs = ['data', 'correction', 'track2', 'track1_curve', 'maxwell2', 'maxwell_reverse', 'no_borders', 'curves', 'curves_track2']
    images_paths = []
    measurements = []
    for dir in data_dirs:
        with open(data_home + dir + '/driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                angle = float(line[3])
                correction = 0.2
                # center camera
                images_paths.append(get_filename(line[0], dir))
                measurements.append(angle)
                # left camera
                images_paths.append(get_filename(line[1], dir))
                measurements.append(angle + correction)
                # right camera
                images_paths.append(get_filename(line[2], dir))
                measurements.append(angle - correction) 
    return (np.array(images_paths), np.array(measurements))

def get_filename(path,dir):
    filename = path.split('/')[-1]
    return (data_home + dir + '/IMG/' + filename)

def load_images(files):
    X_train = []
    for filename in files:
        X_train.append(cv.imread(filename))
    return np.array(X_train)

def flip_images(images, measurements):
    new_images = np.copy(images)
    new_measurements = np.copy(measurements)
    for i in range(new_images.shape[0]):
        new_images[i] = np.flip(images[i],1)
        new_measurements[i] = -measurements[i]
    return (new_images, new_measurements) 