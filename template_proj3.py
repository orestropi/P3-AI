from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import tensorflow as tf
import numpy as np
from tensorflow.keras import initializers
from array import *
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
from tensorflow._api.v2 import math

images = np.load("images.npy")
print(images.shape)
#images.resize((1,784))
print(images)
labels = np.load("labels.npy")
print(labels.shape)

print(labels)

#stratified sampling
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

for label in range(labels.shape[0]):

   array_label = tf.keras.utils.to_categorical(labels[label], num_classes=10, dtype="float32")
   # print(label)
   # print("label", labels[label])
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


#print(len(filtered1))
#print(filtered1.shape)
#print(images.shape)
#print(labels.shape)

train1_length = round(len(filtered1)*.59) #gets 60% of array
valid1_length = round(len(filtered1)*.14) #gets 15% of array
test1_length = round(len(filtered1)*.24) #gets 15% of array
x_train = filtered0[:train1_length] #stores 60% of array
y_train = filteredl0[:train1_length]
x_test = filtered0[train1_length:train1_length+test1_length] #stores 15% or array
y_test = filteredl0[train1_length:train1_length+test1_length] #rest of the array goes into test, should be around 25%
x_val = filtered0[train1_length + test1_length:train1_length+valid1_length + test1_length]
y_val = filteredl0[train1_length + test1_length:train1_length+valid1_length + test1_length]



x_train.extend(filtered1[:train1_length]) #stores 60% of array
y_train.extend(filteredl1[:train1_length])
x_test.extend(filtered1[train1_length:train1_length+test1_length]) #stores 15% or array
y_test.extend(filteredl1[train1_length:train1_length+test1_length]) #rest of the array goes into test, should be around 25%
x_val.extend(filtered1[train1_length + test1_length:train1_length+valid1_length + test1_length])
y_val.extend(filteredl1[train1_length + test1_length:train1_length+valid1_length + test1_length])


x_train.extend(filtered2[:train1_length]) #stores 60% of array
y_train.extend(filteredl2[:train1_length])
x_test.extend(filtered2[train1_length:train1_length+test1_length]) #stores 15% or array
y_test.extend(filteredl2[train1_length:train1_length+test1_length]) #rest of the array goes into test, should be around 25%
x_val.extend(filtered2[train1_length + test1_length:train1_length+valid1_length + test1_length])
y_val.extend(filteredl2[train1_length + test1_length:train1_length+valid1_length + test1_length])#rest of the array goes into test, should be around 25%


x_train.extend(filtered3[:train1_length]) #stores 60% of array
y_train.extend(filteredl3[:train1_length])
x_test.extend(filtered3[train1_length:train1_length+test1_length]) #stores 15% or array
y_test.extend(filteredl3[train1_length:train1_length+test1_length]) #rest of the array goes into test, should be around 25%
x_val.extend(filtered3[train1_length + test1_length:train1_length+valid1_length + test1_length])
y_val.extend(filteredl3[train1_length + test1_length:train1_length+valid1_length + test1_length])


x_train.extend(filtered4[:train1_length]) #stores 60% of array
y_train.extend(filteredl4[:train1_length])
x_test.extend(filtered4[train1_length:train1_length+test1_length]) #stores 15% or array
y_test.extend(filteredl4[train1_length:train1_length+test1_length]) #rest of the array goes into test, should be around 25%
x_val.extend(filtered4[train1_length + test1_length:train1_length+valid1_length + test1_length])
y_val.extend(filteredl4[train1_length + test1_length:train1_length+valid1_length + test1_length])


x_train.extend(filtered5[:train1_length]) #stores 60% of array
y_train.extend(filteredl5[:train1_length])
x_test.extend(filtered5[train1_length:train1_length+test1_length]) #stores 15% or array
y_test.extend(filteredl5[train1_length:train1_length+test1_length]) #rest of the array goes into test, should be around 25%
x_val.extend(filtered5[train1_length + test1_length:train1_length+valid1_length + test1_length])
y_val.extend(filteredl5[train1_length + test1_length:train1_length+valid1_length + test1_length])


x_train.extend(filtered6[:train1_length]) #stores 60% of array
y_train.extend(filteredl6[:train1_length])
x_test.extend(filtered6[train1_length:train1_length+test1_length]) #stores 15% or array
y_test.extend(filteredl6[train1_length:train1_length+test1_length]) #rest of the array goes into test, should be around 25%
x_val.extend(filtered6[train1_length + test1_length:train1_length+valid1_length + test1_length])
y_val.extend(filteredl6[train1_length + test1_length:train1_length+valid1_length + test1_length])


x_train.extend(filtered7[:train1_length]) #stores 60% of array
y_train.extend(filteredl7[:train1_length])
x_test.extend(filtered7[train1_length:train1_length+test1_length]) #stores 15% or array
y_test.extend(filteredl7[train1_length:train1_length+test1_length]) #rest of the array goes into test, should be around 25%
x_val.extend(filtered7[train1_length + test1_length:train1_length+valid1_length + test1_length])
y_val.extend(filteredl7[train1_length + test1_length:train1_length+valid1_length + test1_length])

x_train.extend(filtered8[:train1_length]) #stores 60% of array
y_train.extend(filteredl8[:train1_length])
x_test.extend(filtered8[train1_length:train1_length+test1_length]) #stores 15% or array
y_test.extend(filteredl8[train1_length:train1_length+test1_length]) #rest of the array goes into test, should be around 25%
x_val.extend(filtered8[train1_length + test1_length:train1_length+valid1_length + test1_length])
y_val.extend(filteredl8[train1_length + test1_length:train1_length+valid1_length + test1_length])

x_train.extend(filtered9[:train1_length]) #stores 60% of array
y_train.extend(filteredl9[:train1_length])
x_test.extend(filtered9[train1_length:train1_length+test1_length]) #stores 15% or array
y_test.extend(filteredl9[train1_length:train1_length+test1_length]) #rest of the array goes into test, should be around 25%
x_val.extend(filtered9[train1_length + test1_length:train1_length+valid1_length + test1_length])
y_val.extend(filteredl9[train1_length + test1_length:train1_length+valid1_length + test1_length])



#
# two = images[filtered2]
# three = images[filtered3]
# four = images[filtered4]
# five = images[filtered5]
# six = images[filtered6]
# seven = images[filtered7]
# eight = images[filtered8]
# nine = images[filtered9]
# zero = images[filtered0]
#


# Model Template

model = Sequential() # declare model
model.add(Dense(10, input_shape=(28*28, ), kernel_initializer='he_normal')) # first layer
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu',
    kernel_initializer=initializers.RandomNormal(stddev=0.01),
    bias_initializer=initializers.Zeros()))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax', kernel_initializer=initializers.RandomNormal(stddev=0.01),
    bias_initializer=initializers.Zeros()))
#
#
#
# Fill in Model Here
#
#
model.add(Dense(10, kernel_initializer='he_normal')) # last layer
model.add(Activation('softmax'))


# Compile Model
model.compile(optimizer='sgd',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

# Train Model
history = model.fit(np.array(x_train), np.array(y_train),
                   validation_data = (np.array(x_val), np.array(y_val)),
                   epochs=100,
                   batch_size=512)


# Report Results

print(history.history)
#x_test = tf.stack(x_test, axis=0)
print("Predictions:")
aPredictionArr = model.predict(np.array(x_test))
#np.set_printoptions(threshold=np.inf)
print(aPredictionArr)
#confusion_matrix = tf.math.confusion_matrix(labels=y_test, predictions= aPredictionArr).numpy()
con_matrix = tf.math.confusion_matrix([np.argmax(t) for t in y_test], [np.argmax(p) for p in aPredictionArr])
print("confusion matrix:")
print(con_matrix)