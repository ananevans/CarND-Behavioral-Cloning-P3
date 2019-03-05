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
[image5]: ./writeup/hist0.5.png "Steering Angle Distribution Keeping Half of the Small Angles"
[image6]: ./writeup/hist1.0.png "Steering Angle Distribution"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to train the chosen model
* drive.py for driving the car in autonomous mode (provided)
* .h5 containing a trained convolution neural network 
* writeup.md summarizing the results (this document)
* data.py containing the script to load the images and the data augmentation
* train.py containing the script to train a given convolutional network
* dave2.py containing the Dave2 architecture and calls to train it
* dave2_droput.py containing a modified Dave2 architecture and calls to train it
* a recording of autonomous driving on the first track for both architectures
* a recording of autonomous driving on the second track

#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

First, I used the nvidia dave-2 model:

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

To reduce overfitting, I created a second model with a dropout layer after the first convolutional layer and after each fully-connected layer. The rate for all the Dropout layers is 0.2. I also added L2 regularization with parameter 0.01 to all convolutional layers.

cropping2d_1 (Cropping2D)    (None, 80, 320, 3)        0         
lambda_1 (Lambda)            (None, 80, 320, 3)        0         
conv2d_1 (Conv2D)            (None, 38, 158, 24)       1824      
dropout_1 (Dropout)          (None, 38, 158, 24)       0         
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

#### 3. Results

I trained all both models on four different data sets. First, I use only the data from the first track with and without side camera images. Second, I train on all the data with or without side camera images.

I trained four Dave2 models for five epochs keeping all the data generated. 

| Both Tracks | Side Cameras | Training   | Validation  |
| :---:       |    :----:    |       :--- |        :--- |
| No          | No           | 5*77515    | 5*19379     |



#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

The dataset was generated using the provided simulator. First, I used the keyboard to control the car and I observed that the angles recorded were either very close to zero or to the maximum value (left and right). I scrapped all the data and then controlled the car using the mouse. 

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

Here is the distribution of the angle values:

![alt text][image6]

To reduce the number of angles very close to zero, I decided to keep only half of those values. Here is the new distribution:

![alt text][image5]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
