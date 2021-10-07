Project 3: Machine Learning
CS4341
Project 3 - A Term 2021
Due Date:
Project Submission: by Tuesday, Oct. 12th, 2021 at 9:00 pm
Teams of 1, 2 or 3 students:
Students can work on teams of at most  3 students for this project (i.e., students can work alone, in groups of 2 or in groups of 3 students - your choice). Teams can be the same or different to those of Project 2.
 The Team Formation discussion board can be used to find other students looking for a group. Once you form a group, please update the Canvas Project 3 Groups with your group membership.
Files provided:
Program Template  Download Program Template(you must use this template)
Dataset:
images.npy Download images.npy
labels.npy Download labels.npy
Read about the dataset from which the above files come at the Fashion-MNIST dataset page. (Links to an external site.)
Questions? If you have any questions about the project:
re-read the project description below
read postings and replies on the Project 3 discussion board (someone may have asked the same question before)
if you still have a question, please post it on the Project 3 discussion board
come to office hours
Project Description & Dataset
![image](https://user-images.githubusercontent.com/73619173/136471468-c20c9053-5142-4cb7-97a6-6f8857cc5672.png)
Image taken from the Fashion-MNIST dataset webpage (Links to an external site.) 

Primary Goal:

In this project you will build Artificial Neural Networks (ANNs) for categorizing images. You will write a program that takes an image of a fashion item (like the images above) and outputs what fashion item is represented by the image.

Dataset:

Fashion-MNIST (Links to an external site.) is a dataset of Zalando (Links to an external site.)'s article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. In this project, you will use just a subset of the data that we have randomly selected: There are 6500 images  Download 6500 imagesin our version of the database and 6500 corresponding labels  Download 6500 corresponding labels. Each example is a 28x28 grayscale image, associated with a label from 10 classes.  The 28x28 image matrices contain numbers ranging from 0 (corresponding to white pixels) to 255 (corresponding to black pixels). Labels for each image are also provided with integer values ranging from 0 to 9, corresponding to the fashion item in the image as shown below:

Label	Description
0	T-shirt/top
1	Trouser
2	Pullover
3	Dress
4	Coat
5	Sandal
6	Shirt
7	Sneaker
8	Bag
9	Ankle boot
Taken from the Fashion-MNIST dataset webpage (Links to an external site.). Examples of each label (or class) are shown in 3 rows of the Fashion-MNIST figure shown above. 

 

Tasks:

In this project you will experiment with training machine learning models for identifying which fashion item is represented by a Fashion-MNIST image. You will then submit a report describing your experimental methods and results. You will use a package called Keras (Links to an external site.) implemented in Python3. Keras is an API for TensorFlow. The input to your model will be an image, and the output will be a classification of the fashion item in the image, from 0 to 9. You’ll get a chance to work with common machine learning packages used in state-of-the-art research. In addition, you will practice using the numerical computation package Numpy for preprocessing.

Project Requirements
Your project submission must include a written report and code for training and visualization. Details of what to include are listed below:

Written Report

Model & Training Procedure Description
Include a section describing the set of experiments that you performed, what ANN architectures you experimented with (i.e., number of layers, number of neurons in each layer), what hyperparameters you varied (e.g., number of epochs of training, (mini-)batch size and any other hyperparameter values, weight initialization scheme, activation function), and what accuracies you obtained on each of these experiments

Model Performance & Confusion Matrix
Include a section describing in more detail the most accurate model you were able to obtain: the ANN architecture of your model, including number of layers, number of neurons in each layer, weight initialization scheme, activation function, number of epochs used for training, and batch size used for training.

Include a confusion matrix showing results of testing the model on the test set. The matrix should be a 10-by-10 grid showing which categories images were classified as. Use your confusion matrix to additionally report precision & recall for each of the 10 classes, as well as overall accuracy of your model.

Training Performance Plot
For your best performing ANN, include a plot showing how training accuracy and validation accuracy change over time during training. Graph number of training epochs (x-axis) versus training set and validation set accuracy (y-axis). Hence, your plot should contain two curves.

Visualization
Include 3 visualizations of images that were misclassified by your best performing model and any observations about why you think these images were misclassified.  You will have to create or use a visualization program that takes a 28-by-28 matrix input and translate it into a black-and-white image.

 

Code:

Model Code
Please turn in your preprocessing, model creation, model training, plotting, and confusion matrix code.

 

Model:

Copy of Best Performing Model:
Turn in a copy of your best model saved as `best_trained_model.<ext>' . You need to do “whole-model saving (configuration + weights)” using TensorFlow SavedModel format (Links to an external site.).  Please see more information at Keras’ methods for saving your model (Links to an external site.).

Project Preparatory Tasks and Guidelines
Below are important guidelines to follow for implementing the project using ANNs. A model template  Download templateis provided to you, and these guidelines follow the structure of the template.

 

1) Installing Software and Dependencies

template.py  Download template.pyis written with the Keras API in a Python3 script. You will use this template to build and train a model. To do so, you will need to implement the project in Python3 and install Keras (Links to an external site.) and its dependencies. Please make sure you have a working version of Python3 and Keras as soon as possible, as these programs are necessary for completing the project. For questions about Keras, you can check their FAQ page (Links to an external site.).

 

2) Downloading Data

Raw data is provided and can be downloaded from the following links:

The images.npy  Download images.npyfile contains 6500 images from the Fashion-MNIST dataset.
The labels.npy  Download labels.npycontains the 6500 corresponding labels for the image data.
 

3) Preprocessing Data

 All data is provided as NumPy .npy files. To load and preprocess data, use Python’s  NumPy (Links to an external site.) package

Image data is provided as 28-by-28 matrices of integer pixel values. However, the input to the network will be a flat vector of length 28*28 = 784. You will have to flatten each matrix to be a vector, as illustrated by the toy example below:

![image](https://user-images.githubusercontent.com/73619173/136471585-5b2b97d8-3704-4090-a3f9-2780e3726528.png)

The label for each image is provided as an integer in the range of 0 to 9. However, the output of the network should be structured as a “one-hot vector” of length 10 encoded as follows:

 

![image](https://user-images.githubusercontent.com/73619173/136471646-9375fbbc-00a9-420e-9cfd-d299ad292975.png)

To preprocess data, use NumPy (Links to an external site.)  functions like reshape (Links to an external site.) for changing matrices into vectors. You can also use Keras’s to_categorical (Links to an external site.) function for converting label numbers into one-hot encodings.

After preprocessing, you will need to take your data and randomly split it into Training, Validation, and Test Sets. In order create the three sets of data, use stratified sampling, so that each set contains the same relatively frequency of the ten classes.

You are given 6500 images and labels. The training set should contain ~60% of the data, the validation set should contain ~15% of the data, and the test set should contain ~25% of the data.

Example Stratified Sampling Procedure:

Take data and separate it into 10 classes, one for each digit
From each class:
take 60% at random and put into the Training Set,
take 15% at random and put into the Validation Set,
take the remaining 25% and put into the Test Set
 

3) Building a Model

![image](https://user-images.githubusercontent.com/73619173/136471673-80e8cd69-2f31-443c-a5c8-f8a076d51500.png)
 

In Keras, Models are instantiations of the class Sequential. A Keras model template  Download templatewritten with the Sequential Model API (Links to an external site.) is provided which must be used as a starting point for building your model. The template includes a sample first input layer and output layer. You must limit yourself to “Dense” layers - Keras’ version of traditional fully-connected neural network layers. This portion of the project will involve experimentation. 

Important: Good guidelines for model creation are:

Initialize weights randomly for every layer, try different initialization schemes.
Experiment with using ReLu Activation Units, as well as SeLu and Tanh.
Experiment with number of layers and number of neurons in each layer, including the first layer.
Leave the final layer as it appears in the template with a softmax activation unit.

 

4) Compiling a Model

![image](https://user-images.githubusercontent.com/73619173/136471687-7bf65e2c-a057-4993-b6f6-09a429c5f690.png)

Prior to training a model, you must specify what your loss function for the model is and what your gradient descent method is. Please use the standard categorical cross-entropy and stochastic gradient descent (‘sgd’) when compiling your model (as provided in the template).

  

5) Training a Model

![image](https://user-images.githubusercontent.com/73619173/136471708-d02ff533-473a-4a21-aa4f-ca6b730ee54f.png)

You have the option of changing how many epochs to train your model for and how large your mini-batch size is. Experiment to see what works best. Also remember to include your validation data in the fit() method.

 

6) Reporting Your Results

![image](https://user-images.githubusercontent.com/73619173/136471729-9bed2e78-331c-44c6-aa8d-d6200bc38d87.png)

fit() returns data about your training experiment. In the template this is stored in the “history” variable. Use this information to construct your graph that shows how validation and training accuracy change after every epoch of training.

![image](https://user-images.githubusercontent.com/73619173/136471750-56ffb18c-821c-4eb5-9e60-cf21fb4fdee8.png)

Use the predict() (Links to an external site.) method on model to evaluate what labels your model predicts on test set. Use these and the true labels to construct your confusion matrix, like the toy example below, although you do not need to create a fancy visualization. Your matrix should have 10 rows and 10 columns.

![image](https://user-images.githubusercontent.com/73619173/136471762-9da2177f-0ecc-40b7-ad4b-848fba5fdd0b.png)

Grading Rubric
See detailed descriptions of what each of these parts should contain in Section 2, Project Requirements, above.

Written Report

Description of the set of experiments that you performed: 20 pts
For your best performing model:
Description of the model and the model training procedure: 10 pts
Training performance plot: 10 pts
Performance (accuracy, precision and recall) of your best performing model: 5 pts
Confusion matrix of your best performing model: 5 pts
Visualization of three misclassified images: 10 pts
 Code:

Preprocessing, model creation, model training, plotting and confusion matrix code: 30 pts
Best Performing Model:

Copy of your best performing model: 10 pts
 Total Points: 100 pts + up to 5 bonus points for high accuracy of your best performing model.

  Note: We’ll run your best performing model to determine its accuracy on our own test set.

LATE SUBMISSION POLICY:


Project 3 submission deadline is Tuesday Oct. 12th at 9 pm.

Late submissions will be accepted with a 2*H points penalty, where H is the numbers of hours (or fraction of an hour) you submit your project after the deadline:
For example, if you submit your project after the deadline but between 9:01 pm and 10:00 pm, then you'll get a 2 point penalty; if you submit between 10:01 pm and 11:00 pm you'll get a 4 point penalty; if you submit between 11:01 pm and 12:00 midnight, you get a 6 point penalty and so on (adding 2 points per hour).
No submissions will be accepted after Wed Oct 13th at 9 pm.
Once that your project submission is graded, 2*H points will be taking off from your score. For example, if you submit your project within 4 hours after the deadline and your project score is say 92 points, then your project 2's initial submission grade will be 92 - (4*2) = 84.

 
