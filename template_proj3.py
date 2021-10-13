from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import tensorflow as tf #used to make confusion matrix and add to model
import numpy as np
from tensorflow.keras import initializers #used for model creation
from PIL import Image as im #used to convert matrix into black and white image
from matplotlib import pyplot as plt #used to plot the accuracies

#commented out now, just used to make sure our data was splitting correctly

#testing splitting data
#our_array = np.array([4,5,6,7,8])
#print(our_array)
#rng = np.random.default_rng()
#rng.shuffle(our_array)
#print(our_array)
#train_length = round(len(our_array)*.6)
#print(train_length)
#our_array = np.delete(our_array, np.arange(train_length))
#print(our_array)

#preprocessing
images = np.load("images.npy") #loading images
#print(images.shape)
#print(images)
labels = np.load("labels.npy") #loading corresponding labels
#print(labels.shape)
#print(labels)

#stratified sampling

#initializing arrays for sorting by label, the images will be stored here
filtered0 = []
filtered1 = []
filtered2 = []
filtered3 = []
filtered4 = []
filtered5 = []
filtered6 = []
filtered7 = []
filtered8 = []
filtered9 = []

#initializing arrays for sorting by label, labels will be stored here
filteredl0 = []
filteredl1 = []
filteredl2 = []
filteredl3 = []
filteredl4 = []
filteredl5 = []
filteredl6 = []
filteredl7 = []
filteredl8 = []
filteredl9 = []

#go through every label and sort itself and the corresponding image and place in correct array
for label in range(labels.shape[0]):
    #change the labels from int to array of zeros and one
    array_label = tf.keras.utils.to_categorical(labels[label], num_classes=10, dtype="float32")
    #check the int value of the label and place corresponding array label and image into correct array
    if labels[label] == 0:
        filtered0.append(images[label])
        filteredl0.append(array_label);
    elif labels[label] == 1:
        filtered1.append(images[label])
        filteredl1.append(array_label);
    elif labels[label] == 2:
        filtered2.append(images[label])
        filteredl2.append(array_label);
    elif labels[label] == 3:
        filteredl3.append(array_label);
        filtered3.append(images[label])
    elif labels[label] == 4:
        filtered4.append(images[label])
        filteredl4.append(array_label);
    elif labels[label] == 5:
        filtered5.append(images[label])
        filteredl5.append(array_label);
    elif labels[label] == 6:
        filtered6.append(images[label])
        filteredl6.append(array_label);
    elif labels[label] == 7:
        filtered7.append(images[label])
        filteredl7.append(array_label);
    elif labels[label] == 8:
        filtered8.append(images[label])
        filteredl8.append(array_label);
    else:
        filtered9.append(images[label])
        filteredl9.append(array_label);

#splitting the data into test, valid, and train sets

#go through the images with label 0 and take 60% and put in training, 15% in valid, and 25% in test set
train1_length = round(len(filtered1)*.59) #gets the length of 60% of array
valid1_length = round(len(filtered1)*.14) #gets the length of 15% of array
test1_length = round(len(filtered1)*.24) #gets the length of 25% of array
x_train = filtered0[:train1_length] #stores 60% of array images for training set
y_train = filteredl0[:train1_length] #stores 60% or array labels for training set
x_test = filtered0[train1_length:train1_length+test1_length] #stores 25% of array images for test set
y_test = filteredl0[train1_length:train1_length+test1_length] #stores 25% of array labels for test set
x_val = filtered0[train1_length + test1_length:train1_length+valid1_length + test1_length] #stores 15% of array images for valid set
y_val = filteredl0[train1_length + test1_length:train1_length+valid1_length + test1_length] #stores 15% of array labels for valid set

#go through the images with label 1 and take 60% and put in training, 15% in valid, and 25% in test set
x_train.extend(filtered1[:train1_length])
y_train.extend(filteredl1[:train1_length])
x_test.extend(filtered1[train1_length:train1_length+test1_length])
y_test.extend(filteredl1[train1_length:train1_length+test1_length])
x_val.extend(filtered1[train1_length + test1_length:train1_length+valid1_length + test1_length])
y_val.extend(filteredl1[train1_length + test1_length:train1_length+valid1_length + test1_length])

#go through the images with label 2 and take 60% and put in training, 15% in valid, and 25% in test set
x_train.extend(filtered2[:train1_length])
y_train.extend(filteredl2[:train1_length])
x_test.extend(filtered2[train1_length:train1_length+test1_length])
y_test.extend(filteredl2[train1_length:train1_length+test1_length])
x_val.extend(filtered2[train1_length + test1_length:train1_length+valid1_length + test1_length])
y_val.extend(filteredl2[train1_length + test1_length:train1_length+valid1_length + test1_length])

#go through the images with label 3 and take 60% and put in training, 15% in valid, and 25% in test set
x_train.extend(filtered3[:train1_length])
y_train.extend(filteredl3[:train1_length])
x_test.extend(filtered3[train1_length:train1_length+test1_length])
y_test.extend(filteredl3[train1_length:train1_length+test1_length])
x_val.extend(filtered3[train1_length + test1_length:train1_length+valid1_length + test1_length])
y_val.extend(filteredl3[train1_length + test1_length:train1_length+valid1_length + test1_length])

#go through the images with label 4 and take 60% and put in training, 15% in valid, and 25% in test set
x_train.extend(filtered4[:train1_length])
y_train.extend(filteredl4[:train1_length])
x_test.extend(filtered4[train1_length:train1_length+test1_length])
y_test.extend(filteredl4[train1_length:train1_length+test1_length])
x_val.extend(filtered4[train1_length + test1_length:train1_length+valid1_length + test1_length])
y_val.extend(filteredl4[train1_length + test1_length:train1_length+valid1_length + test1_length])

#go through the images with label 5 and take 60% and put in training, 15% in valid, and 25% in test set
x_train.extend(filtered5[:train1_length])
y_train.extend(filteredl5[:train1_length])
x_test.extend(filtered5[train1_length:train1_length+test1_length])
y_test.extend(filteredl5[train1_length:train1_length+test1_length])
x_val.extend(filtered5[train1_length + test1_length:train1_length+valid1_length + test1_length])
y_val.extend(filteredl5[train1_length + test1_length:train1_length+valid1_length + test1_length])

#go through the images with label 6 and take 60% and put in training, 15% in valid, and 25% in test set
x_train.extend(filtered6[:train1_length])
y_train.extend(filteredl6[:train1_length])
x_test.extend(filtered6[train1_length:train1_length+test1_length])
y_test.extend(filteredl6[train1_length:train1_length+test1_length])
x_val.extend(filtered6[train1_length + test1_length:train1_length+valid1_length + test1_length])
y_val.extend(filteredl6[train1_length + test1_length:train1_length+valid1_length + test1_length])

#go through the images with label 7 and take 60% and put in training, 15% in valid, and 25% in test set
x_train.extend(filtered7[:train1_length])
y_train.extend(filteredl7[:train1_length])
x_test.extend(filtered7[train1_length:train1_length+test1_length])
y_test.extend(filteredl7[train1_length:train1_length+test1_length])
x_val.extend(filtered7[train1_length + test1_length:train1_length+valid1_length + test1_length])
y_val.extend(filteredl7[train1_length + test1_length:train1_length+valid1_length + test1_length])

#go through the images with label 8 and take 60% and put in training, 15% in valid, and 25% in test set
x_train.extend(filtered8[:train1_length])
y_train.extend(filteredl8[:train1_length])
x_test.extend(filtered8[train1_length:train1_length+test1_length])
y_test.extend(filteredl8[train1_length:train1_length+test1_length])
x_val.extend(filtered8[train1_length + test1_length:train1_length+valid1_length + test1_length])
y_val.extend(filteredl8[train1_length + test1_length:train1_length+valid1_length + test1_length])

#go through the images with label 8 and take 60% and put in training, 15% in valid, and 25% in test set
x_train.extend(filtered9[:train1_length])
y_train.extend(filteredl9[:train1_length])
x_test.extend(filtered9[train1_length:train1_length+test1_length])
y_test.extend(filteredl9[train1_length:train1_length+test1_length])
x_val.extend(filtered9[train1_length + test1_length:train1_length+valid1_length + test1_length])
y_val.extend(filteredl9[train1_length + test1_length:train1_length+valid1_length + test1_length])


# Model Template

model = Sequential() # declare model
model.add(Dense(45, input_shape=(28*28, ), kernel_initializer='lecun_normal')) # first layer

#model filled in here
model.add(Activation('relu'))
model.add(Dense(640, activation='tanh',kernel_initializer=initializers.RandomNormal(stddev=0.0001)))
model.add(tf.keras.layers.Dense(400, kernel_initializer='lecun_normal',activation='tanh'))
model.add(Dropout(.05))

model.add(Dense(10, kernel_initializer='he_normal')) # last layer
model.add(Activation('softmax'))


# Compile Model
model.compile(optimizer='sgd',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

# Train Model
history = model.fit(np.array(x_train), np.array(y_train),
                   validation_data = (np.array(x_val), np.array(y_val)),
                   epochs=200,
                   batch_size=300)


# Report Results

print(history.history)

#printing out our predictions that we get when we run the test images on our model
print("Predictions:")
aPredictionArr = model.predict(np.array(x_test)) #gets a prediction for each image in the form of an array
print(aPredictionArr) #prints these predictions out

#confusion_matrix creation
con_matrix = tf.math.confusion_matrix([np.argmax(t) for t in y_test], [np.argmax(p) for p in aPredictionArr])
print("Confusion matrix:") #printing the confusion matrix
print(con_matrix)

#saving 3 images that are incorrect predictions
count = 0
#go through the predictions
for y in range(len(aPredictionArr)):
    if count < 3: #only get 3 images
        prediction_index = np.argmax(aPredictionArr[y]) #gets index of predicted value
        actual_index = np.argmax(y_test[y]) #gets index of actual value
        if prediction_index != actual_index: #checks if the index for the predicted values are the same
            array = np.reshape(x_test[y], (28,28)) #reshape array to 28x28
            data = im.fromarray(array) #create image
            count = count + 1
            data.save('Wrong Prediction'+str(count)+'.png') #save image
    else:
        break

#prints the accuracy of the entire model
print("Total accuracy:")
totalRight = 0
for i in range(0, 10):
    totalRight += con_matrix[i][i] #counts the number of predictions on the diagonal
print(totalRight/1560) #total number of diagonal predicions divided by total number of images (gives a percent)

#save model
model.save('project_3_model')

#plotting
plt.plot(history.history['accuracy']) #training accuracy data
plt.plot(history.history['val_accuracy']) #validation set data
plt.title('Training and Validation Accuracy over Time') #plot title
plt.ylabel('Accuracy') #y axis label
plt.xlabel('Number of training epochs') #x axis label
plt.legend(['Training set', 'Validation set'], loc='upper left') #adding a key for the two plots
plt.show() #prints the plot