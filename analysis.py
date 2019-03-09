import data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from fileinput import filename

def data_distribution(keep_straight_rate):
    samples = data.load_data(True, True, keep_straight_rate=keep_straight_rate)
    y_train = samples[:,2]
    y_train = y_train.astype(float)
    plt.hist(y_train, bins=21)
    filename = '/home/ans5k/work/CarND-Behavioral-Cloning-P3/writeup/hist-track1.png'
    plt.savefig( filename, bbox_inches='tight')

def original():
    center = plt.imread('/home/ans5k/work/CarND-Behavioral-Cloning-P3/data/track1_mouse/IMG/center_2019_03_01_11_50_03_923.jpg')
    left = plt.imread('/home/ans5k/work/CarND-Behavioral-Cloning-P3/data/track1_mouse/IMG/left_2019_03_01_11_50_03_923.jpg')
    right = plt.imread('/home/ans5k/work/CarND-Behavioral-Cloning-P3/data/track1_mouse/IMG/right_2019_03_01_11_50_03_923.jpg') 
    gs = gridspec.GridSpec(1,3)
    plt.subplot(gs[0, 0]).imshow(center)
    plt.subplot(gs[0, 1]).imshow(left)
    plt.subplot(gs[0, 2]).imshow(right)
    plt.tight_layout()
    plt.savefig('/home/ans5k/work/CarND-Behavioral-Cloning-P3/writeup/input_images.png', bbox_inches='tight')

def cropping():
    center = plt.imread('/home/ans5k/work/CarND-Behavioral-Cloning-P3/data/track1_mouse/IMG/center_2019_03_01_11_50_03_923.jpg')
    left = plt.imread('/home/ans5k/work/CarND-Behavioral-Cloning-P3/data/track1_mouse/IMG/left_2019_03_01_11_50_03_923.jpg')
    right = plt.imread('/home/ans5k/work/CarND-Behavioral-Cloning-P3/data/track1_mouse/IMG/right_2019_03_01_11_50_03_923.jpg') 
    cropped_center = center[60:140,:]
    cropped_left = left[60:140,:]
    cropped_right = right[60:140,:]
    
    gs = gridspec.GridSpec(1,3)
    plt.subplot(gs[0, 0]).imshow(cropped_center)
    plt.subplot(gs[0, 1]).imshow(cropped_left)
    plt.subplot(gs[0, 2]).imshow(cropped_right)
    plt.tight_layout()
    plt.savefig('/home/ans5k/work/CarND-Behavioral-Cloning-P3/writeup/cropped_images.png', bbox_inches='tight')
    
def flipping():
    center = plt.imread('/home/ans5k/work/CarND-Behavioral-Cloning-P3/data/track1_mouse/IMG/center_2019_03_01_11_50_03_923.jpg')
    left = plt.imread('/home/ans5k/work/CarND-Behavioral-Cloning-P3/data/track1_mouse/IMG/left_2019_03_01_11_50_03_923.jpg')
    right = plt.imread('/home/ans5k/work/CarND-Behavioral-Cloning-P3/data/track1_mouse/IMG/right_2019_03_01_11_50_03_923.jpg') 
    
    gs = gridspec.GridSpec(1,3)
    plt.subplot(gs[0, 0]).imshow(np.flip(center,1))
    plt.subplot(gs[0, 1]).imshow(np.flip(left,1))
    plt.subplot(gs[0, 2]).imshow(np.flip(right,1))
    plt.tight_layout()
    plt.savefig('/home/ans5k/work/CarND-Behavioral-Cloning-P3/writeup/flipped_images.png', bbox_inches='tight')

def augmetation():
    center = plt.imread('/home/ans5k/work/CarND-Behavioral-Cloning-P3/data/track1_mouse/IMG/center_2019_03_01_11_50_03_923.jpg')
    left = plt.imread('/home/ans5k/work/CarND-Behavioral-Cloning-P3/data/track1_mouse/IMG/left_2019_03_01_11_50_03_923.jpg')
    right = plt.imread('/home/ans5k/work/CarND-Behavioral-Cloning-P3/data/track1_mouse/IMG/right_2019_03_01_11_50_03_923.jpg') 
    
    gs = gridspec.GridSpec(1,3)
    plt.subplot(gs[0, 0]).imshow(data.brightness(center))
    plt.subplot(gs[0, 1]).imshow(data.blurr1(center))
    plt.subplot(gs[0, 2]).imshow(data.blurr2(center))
    plt.tight_layout()
    plt.savefig('/home/ans5k/work/CarND-Behavioral-Cloning-P3/writeup/augmented_images.png', bbox_inches='tight')

# original()
# cropping()
# flipping()
# augmetation()

#data_distribution(1.0)
data_distribution(1.0)