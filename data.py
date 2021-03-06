import csv
import numpy as np
import cv2 as cv
import random
import glob

data_home = '/home/ans5k/work/CarND-Behavioral-Cloning-P3/data/'

def load_data(track1, side_cameras, keep_straight_rate = 1.0):
    if track1:
        data_dirs = glob.glob(data_home + 'track1_data*')
        #data_dirs = glob.glob(data_home + 'track1_data*')
    else:
        data_dirs = glob.glob(data_home + '*')
        #data_dirs = glob.glob(data_home + 'track1_data*')
    result = []
    for dir in data_dirs:
        with open(dir + '/driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            keep = random.uniform(0.0,1.0)
            for line in reader:
                angle = float(line[3])
                correction = 0.2
                if abs(angle) > 0.1 or keep <= keep_straight_rate:
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


def brightness(image):
    new_image = cv.cvtColor(image, cv.COLOR_BGR2HLS)
    alpha = random.uniform(0.5, 1.5)
    new_image[:,:,1] = new_image[:,:,1] * alpha 
    new_image[:,:,1][new_image[:,:,1]>255] = 255
    return cv.cvtColor(new_image, cv.COLOR_HLS2BGR)

def blurr1(image):
    return cv.GaussianBlur(image,(5,5),0)

def blurr2(image):
    return cv.medianBlur(image,5)


import sklearn.utils
def generator(samples, batch_size=100, augment_data = True):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        
        if augment_data:
            factor = 5
        else:
            factor = 1
        
        for offset in range(0, num_samples, int(batch_size/factor)):
            batch_samples = samples[offset:offset+int(batch_size/factor)]
            images = []
            angles = []
            for image_name, flip, angle in batch_samples:
                #print(image_name, flip, angle)
                angle = float(angle)
                if flip == 'True':
                    flip = True
                else:
                    flip = False
                #print(image_name, flip, angle)
                image = cv.imread(image_name)
                if flip:
                    image = np.flip(image,1)
                images.append(cv.cvtColor(image, cv.COLOR_BGR2RGB))
                angles.append(np.float(angle))
                if augment_data:
                    images.append(cv.cvtColor(brightness(image), cv.COLOR_BGR2RGB))
                    angles.append(np.float(angle))
                    images.append(cv.cvtColor(brightness(image), cv.COLOR_BGR2RGB))
                    angles.append(np.float(angle))
                    images.append(cv.cvtColor(blurr1(image), cv.COLOR_BGR2RGB))
                    angles.append(np.float(angle))
                    images.append(cv.cvtColor(blurr2(image), cv.COLOR_BGR2RGB))
                    angles.append(np.float(angle))
            X_train = np.array(images)
            y_train = np.array(angles)
            yield (X_train, y_train)


def load_images(samples):
    images = []
    angles = []
    for image_name, flip, angle in samples:
        angle = float(angle)
        if flip == 'True':
            flip = True
        else:
            flip = False
        image = cv.imread(image_name)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        if flip:
            images.append(np.flip(image,1))
        else:
            images.append(image)
        angles.append(angle)
    X_train = np.array(images)
    y_train = np.array(angles)
    return (X_train, y_train)

                    
def get_filename(path,dir):
    filename = path.split('/')[-1]
    return (dir + '/IMG/' + filename)
