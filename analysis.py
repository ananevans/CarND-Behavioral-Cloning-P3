import data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def data_distribution():
    samples = data.load_data(False, True)
    y_train = samples[:,2]
    print(y_train.shape)
    y_train = y_train.astype(float)
    print(y_train.shape[0] - np.count_nonzero(y_train, 0))
 
    plt.hist(y_train, bins=25)
    plt.show()

def cropping():
    center = plt.imread('/home/ans5k/work/CarND-Behavioral-Cloning-P3/data/track1_center/IMG/center_2019_02_27_21_20_15_232.jpg')
    left = plt.imread('/home/ans5k/work/CarND-Behavioral-Cloning-P3/data/track1_center/IMG/left_2019_02_27_21_20_15_232.jpg')
    right = plt.imread('/home/ans5k/work/CarND-Behavioral-Cloning-P3/data/track1_center/IMG/right_2019_02_27_21_20_15_232.jpg') 
    cropped_center = center[55:140,:]
    cropped_left = left[55:140,:]
    cropped_right = right[55:140,:]
    
    gs = gridspec.GridSpec(2,3)
    plt.subplot(gs[0, 0]).imshow(center)
    plt.subplot(gs[0, 1]).imshow(left)
    plt.subplot(gs[0, 2]).imshow(right)
    plt.subplot(gs[1, 0]).imshow(cropped_center)
    plt.subplot(gs[1, 1]).imshow(cropped_left)
    plt.subplot(gs[1, 2]).imshow(cropped_right)
    plt.tight_layout()
    plt.savefig('/home/ans5k/work/CarND-Behavioral-Cloning-P3/writeup/input_images.png', bbox_inches='tight')
    
#     fig, axeslist = plt.subplots(ncols=3, nrows=2)
#     axeslist[0,0].imshow(center)
#     axeslist[0,1].imshow(left)
#     axeslist[0,2].imshow(right)
#     axeslist[1,0].imshow(cropped_center)
#     axeslist[1,1].imshow(cropped_left)
#     axeslist[1,2].imshow(cropped_right)
#     plt.tight_layout()
#     plt.savefig('/home/ans5k/work/CarND-Behavioral-Cloning-P3/writeup/input_images.png', bbox_inches='tight')
    
#cropping()
data_distribution()