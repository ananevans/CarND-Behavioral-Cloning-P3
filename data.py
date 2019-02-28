import csv
import numpy as np
import cv2 as cv
#data_home = '/home/nora/work/CarND-Behavioral-Cloning-P3/data/'
data_home = '/home/ans5k/work/CarND-Behavioral-Cloning-P3/data/'

def load_data(track1, side_cameras):
    if track1:
#         data_dirs = ['data', 'track1_center', 'track1_center_reverse', 'track1_curves', 
#                  'track1_curves_reverse', 'track1_off_center', 'track1_off_center_reverse']
        data_dirs = ['track1_center', 'track1_center_reverse']
    else:
        data_dirs = ['data', 'track1_center', 'track1_center_reverse', 'track1_curves', 
                 'track1_curves_reverse', 'track1_off_center', 'track1_off_center_reverse', 'track2', 'track2_reverse']
    result = []
    for dir in data_dirs:
        with open(data_home + dir + '/driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                angle = float(line[3])
                correction = 0.2
                # center camera
                result.append((get_filename(line[0], dir), False, angle))
                result.append((get_filename(line[0], dir), True, -angle))
                if side_cameras:
                    # left camera
                    result.append((get_filename(line[1], dir), False, (angle + correction)))
                    result.append((get_filename(line[1], dir), True, -(angle + correction)))
                    # right camera
                    result.append((get_filename(line[2], dir), False, (angle - correction)))
                    result.append((get_filename(line[2], dir), True, -(angle - correction)))
    return np.array(result)


import sklearn.utils
def generator(samples, batch_size=10000):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for image_name, flip, angle in batch_samples:
                angle = float(angle)
                flip = bool(flip)
                image = cv.imread(image_name)
                if flip:
                    images.append(np.flip(image,1))
                else:
                    images.append(image)
                angles.append(np.float(angle))
            X_train = np.array(images)
            y_train = np.array(angles)
            yield (X_train, y_train)

                
def load_images(samples):
    images = []
    angles = []
    for image_name, flip, angle in samples:
        angle = float(angle)
        flip = bool(flip)
        image = cv.imread(image_name)
        if flip:
            images.append(np.flip(image,1))
        else:
            images.append(image)
        angles.append(np.float(angle))
    X_train = np.array(images)
    y_train = np.array(angles)
    return (X_train, y_train)

                    

                    
def get_filename(path,dir):
    filename = path.split('/')[-1]
    return (data_home + dir + '/IMG/' + filename)

# def load_images(files):
#     X_train = []
#     for filename in files:
#         X_train.append(cv.imread(filename))
#     return np.array(X_train)
# 
# def flip_images(images, measurements):
#     new_images = np.copy(images)
#     new_measurements = np.copy(measurements)
#     for i in range(new_images.shape[0]):
#         new_images[i] = np.flip(images[i],1)
#         new_measurements[i] = -measurements[i]
#     return (new_images, new_measurements) 
