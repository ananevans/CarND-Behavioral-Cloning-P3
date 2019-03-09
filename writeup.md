# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup/input_images.png "Original Images"
[image2]: ./writeup/cropped_images.png "Cropped Images"
[image3]: ./writeup/flipped_images.png "Flipped Images"
[image4]: ./writeup/augmented_images.png "Augmented Images"
[image5]: ./writeup/hist-all.png "Steering Angle Distribution"
[image6]: ./writeup/hist-track1.png "Steering Angle (track1 only)"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to train the chosen model
* drive.py for driving the car in autonomous mode (provided)
* data.py containing the script to load the images and the data augmentation
* train.py containing the script to train a given convolutional network
* dave2.py containing the Dave2 architecture and calls to train it
* dave2_dropout.py containing a modified Dave2 architecture and calls to train it
* models are available [here](http://www.cs.virginia.edu/~ans5k/CarND-Behavioral-Cloning-P3)
* writeup.md summarizing the results (this document)
* recordings of autonomous driving on both tracks are linked from this document and [here](./models/udacity_and_track_2/)
* recording of the best [run](./models/udacity_and_track_2/dave2_track1.mp4)
* the training data is available [here](https://github.com/ananevans/CarND-Behavioral-Cloning-P3-Data)

#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
Replace model.h5 with one of the models available [here](http://www.cs.virginia.edu/~ans5k/CarND-Behavioral-Cloning-P3).


#### 3. Submission code is usable and readable

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

First, I used the Dave-2 model. The first layer is the normalization suggested in the project directions. The second layer is cropping the image to contain only the relevant road image. Next, there are five convolution layers. These layers are acting as feature extractors. One such example is the network learns to identify the sides of the road even when they are significantly different than the ones in the training daata. After the flattening layer, there are four dense layers, with the last one consisting of one neuron since this is a regression network that output a real value representing the steering angle.

Layer (type)                 Output Shape              Param # 
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
cropping2d_1 (Cropping2D)    (None, 80, 320, 3)        0         
conv2d_1 (Conv2D)            (None, 38, 158, 24)       1824      
conv2d_2 (Conv2D)            (None, 17, 77, 36)        21636     
conv2d_3 (Conv2D)            (None, 7, 37, 48)         43248     
conv2d_4 (Conv2D)            (None, 5, 35, 64)         27712     
conv2d_5 (Conv2D)            (None, 3, 33, 64)         36928     
flatten_1 (Flatten)          (None, 6336)              0         
dense_1 (Dense)              (None, 100)               633700    
dense_2 (Dense)              (None, 50)                5050      
dense_3 (Dense)              (None, 10)                510       
dense_4 (Dense)              (None, 1)                 11        


#### 2. Attempts to reduce overfitting in the model

To reduce overfitting, I created a second model with dropout layers after each fully-connected layer with rate 0.2. I also added L2 kernel regularization with parameter 0.01 to all convolutional layers. I was hoping this model would be easier to train on a training set containing some bad data and might generalize to the second track without adding images from it in the training set. 

Layer (type)                 Output Shape              Param # 
cropping2d_1 (Cropping2D)    (None, 80, 320, 3)        0         
lambda_1 (Lambda)            (None, 80, 320, 3)        0         
conv2d_1 (Conv2D)            (None, 38, 158, 24)       1824      
conv2d_2 (Conv2D)            (None, 17, 77, 36)        21636     
conv2d_3 (Conv2D)            (None, 7, 37, 48)         43248     
conv2d_4 (Conv2D)            (None, 5, 35, 64)         27712     
conv2d_5 (Conv2D)            (None, 3, 33, 64)         36928     
flatten_1 (Flatten)          (None, 6336)              0         
dense_1 (Dense)              (None, 100)               633700    
dropout_2 (Dropout)          (None, 100)               0         
dense_2 (Dense)              (None, 50)                5050      
dropout_3 (Dropout)          (None, 50)                0         
dense_3 (Dense)              (None, 10)                510       
dropout_4 (Dropout)          (None, 10)                0         
dense_4 (Dense)              (None, 1)                 11        

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

I tried different data sets:
1. Udacity data only: both models complete track 1, but fail very quickly on track 2 ([dave2](http://www.cs.virginia.edu/~ans5k/CarND-Behavioral-Cloning-P3/udacity_data_augmentation/dave2.mp4), [dave2-dropout](http://www.cs.virginia.edu/~ans5k/CarND-Behavioral-Cloning-P3/udacity_data_augmentation/dave2-dropout.mp4)). Both models are available [here](http://www.cs.virginia.edu/~ans5k/CarND-Behavioral-Cloning-P3/udacity_data_augmentation/).
2. Udacity data, plus data I recorded on track 2: dave2 model completes both tracks([track 1](http://www.cs.virginia.edu/~ans5k/CarND-Behavioral-Cloning-P3/udacity_and_track_2/dave2_track1.mp4), [track2](http://www.cs.virginia.edu/~ans5k/CarND-Behavioral-Cloning-P3/udacity_and_track_2/dave2_track2.mp4)), but the model with normalization does not.
3. My data for track 1 only: both models complete track 1: [dave2](http://www.cs.virginia.edu/~ans5k/CarND-Behavioral-Cloning-P3/noras_data/dave2-track1.mp4), [dave2-dropout](http://www.cs.virginia.edu/~ans5k/CarND-Behavioral-Cloning-P3/noras_data/dave2-dropout_track1.mp4). The models are available [here](http://www.cs.virginia.edu/~ans5k/CarND-Behavioral-Cloning-P3/noras_data/).
4. All data: the dave2 model completes both tracks ([track1](http://www.cs.virginia.edu/~ans5k/CarND-Behavioral-Cloning-P3/noras_data/dave2-all-track1.mp4), [track2](http://www.cs.virginia.edu/~ans5k/CarND-Behavioral-Cloning-P3/noras_data/dave2-all-track2.mp4)), the dave2 model with dropout fails on both tracks. The models are available [here](http://www.cs.virginia.edu/~ans5k/CarND-Behavioral-Cloning-P3/noras_data/).

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

Since we do not work with a benchmark data set, I chose to start by using a well-known network, the NVidia Dave-2. This network has no regularization, and I thought that adding regularization to all layers, it would make the network easier to train on the data I generated which was not so smooth as the provided one. 

First, I use the keyboard to generate my own data and did not used the data provided. Both models failed to complete the first track. The histogram showed that the steering angles were either zero or maximum left or right. I tried to use the mouse and generated a set of data at very low speed and very carefully. The models trained on this data failed even worse. Next, I tried to generate data driving fast. The models trained on this data did a little better, but still failed. I noticed then, the test is performed at constant medium speed. I tried to generate the data close to that speed and it almost worked. There were two places were there were still problems. First, right after the bridge the car was going off the road, because there is no shoulder. I collected more data driving by that spot and it solved the problem. The second problem was around the last sharp curve. In the training, towards the end of the track, I kept forgeting to check the speed and I was taking it too fast. Again, I added more data driving carefully and kept the car on the road. 

Since I had all this problems with data collection, I decided to experiment using the data provided by Udacity. The Dave-2 network is pretty big and the data provided not a lot. I decided to add data augmentation to get a training data set five times bigger. This approach seemed to improve the smoothness of the driving.

Since the network with strong regularization did not generalize to the second track and creates a worse model on the not so smooth data, I decided it was not an useful approach.

By looking at the steering angles, as suggested in the directions, there are many more examples with close to zero steering angles. To reduce the weight of such examples in the training set, I decided not to flip the images with angles close to zero. For all the other angles, I flip the image and assign the negative of the steering angle. This strategy doubles the available training data for bigger absolute value steering angles. 

I attempted to further reduce the number of examples with small steering angles by randomly dropping with a given probability images from the overrepresented set. This strategy created a nice histogram, but made the model learn to steer too much. I abandoned this strategy.


#### 2. Final Model Architecture

The smallest loss is realized by the Dave-2 architecture trained with data provided by Udacity for track 1 and my own data for track 2.

#### 3. Creation of the Training Set & Training Process

The dataset was generated using the provided simulator. First, I used the keyboard to control the car and I observed that the angles recorded were either very close to zero or to the maximum value (left and right). I scrapped all the data and then controlled the car using the mouse. Even with the mouse, if the speed of the car is not close to 9Mph (the speed of the car in the test) the data collected is bad. 

I recorded the following sets of data:
* three laps driving around the first track
* three laps driving around the first track in the opposite direction
* one lap driving around the second track
* one lap driving around the second track in the opposite direction
* carefully driving two-three times around the curves of both tracks in both directions
* starting from an off-center position driving to the center

Here is an example image of center lane driving recorded with the center, left and right cameras:

![alt text][image1]

To eliminate the irrelevant information, I cropped the pictures as suggested in the project directions. Here are the cropped images:

![alt text][image2]

To augment the data sat, I also flipped images and angles if the angle is not very close to zero. I flipped 

![alt text][image3]

I also augment the input data by radomly changing the brightness (file data.py, lines 37-42) and applying Gaussian (file data.py, line 45)  and median blurring(file data.py, line 48).

![alt text][image4]

Here is the distribution of the angle values for all data:

![alt text][image5]

Here is the distribution of the angle values for my track 1 data:

![alt text][image6]

From this two images, we can observe that adding images from the second track was very useful in improving the distribution of steering angles.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I experimented with keeping only a fraction of angles close to zero, but it didn't seem to work. I do not flip the image if the angle is close to zero.

### Conclusions

The dave2 model with dropout is harder to train with data recorded from less than ideal runs. The use of side images was very important. Without those images the training set was three times smaller and had too many values close to zero.

