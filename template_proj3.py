from keras.models import Sequential
from keras.layers import Dense, Activation
import tensorflow as tf
import numpy as np

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
array_labels = tf.keras.utils.to_categorical(labels, num_classes=None)
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
    # print(label)
    # print("label", labels[label])
    if labels[label] == 0:
        filtered0.append(images[label])
        filteredl0.append(array_labels[label]);
    elif labels[label] == 1:
        filtered1.append(images[label])
        filteredl1.append(array_labels[label]);
    elif labels[label] == 2:
        filtered2.append(images[label])
        filteredl2.append(array_labels[label]);
    elif labels[label] == 3:
        filteredl3.append(array_labels[label]);
        filtered3.append(images[label])
    elif labels[label] == 4:
        filtered4.append(images[label])
        filteredl4.append(array_labels[label]);
    elif labels[label] == 5:
        filtered5.append(images[label])
        filteredl5.append(array_labels[label]);
    elif labels[label] == 6:
        filtered6.append(images[label])
        filteredl6.append(array_labels[label]);
    elif labels[label] == 7:
        filtered7.append(images[label])
        filteredl7.append(array_labels[label]);
    elif labels[label] == 8:
        filtered8.append(images[label])
        filteredl8.append(array_labels[label]);
    else:
        filtered9.append(images[label])
        filteredl9.append(9);


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



x_train += filtered1[:train1_length] #stores 60% of array
y_train += filteredl1[:train1_length]
x_test += filtered1[train1_length:train1_length+test1_length] #stores 15% or array
y_test += filteredl1[train1_length:train1_length+test1_length] #rest of the array goes into test, should be around 25%
x_val += filtered1[train1_length + test1_length:train1_length+valid1_length + test1_length]
y_val += filteredl1[train1_length + test1_length:train1_length+valid1_length + test1_length]


x_train += filtered2[:train1_length] #stores 60% of array
y_train += filteredl2[:train1_length]
x_test += filtered2[train1_length:train1_length+test1_length] #stores 15% or array
y_test += filteredl2[train1_length:train1_length+test1_length] #rest of the array goes into test, should be around 25%
x_val += filtered2[train1_length + test1_length:train1_length+valid1_length + test1_length]
y_val += filteredl2[train1_length + test1_length:train1_length+valid1_length + test1_length] #rest of the array goes into test, should be around 25%


x_train += filtered3[:train1_length] #stores 60% of array
y_train += filteredl3[:train1_length]
x_test += filtered3[train1_length:train1_length+test1_length] #stores 15% or array
y_test += filteredl3[train1_length:train1_length+test1_length] #rest of the array goes into test, should be around 25%
x_val += filtered3[train1_length + test1_length:train1_length+valid1_length + test1_length]
y_val += filteredl3[train1_length + test1_length:train1_length+valid1_length + test1_length] #rest of the array goes into test, should be around 25%


x_train += filtered4[:train1_length] #stores 60% of array
y_train += filteredl4[:train1_length]
x_test += filtered4[train1_length:train1_length+test1_length] #stores 15% or array
y_test += filteredl4[train1_length:train1_length+test1_length] #rest of the array goes into test, should be around 25%
x_val += filtered4[train1_length + test1_length:train1_length+valid1_length + test1_length]
y_val += filteredl4[train1_length + test1_length:train1_length+valid1_length + test1_length] #rest of the array goes into test, should be around 25%


x_train += filtered5[:train1_length] #stores 60% of array
y_train += filteredl5[:train1_length]
x_test += filtered5[train1_length:train1_length+test1_length] #stores 15% or array
y_test += filteredl5[train1_length:train1_length+test1_length] #rest of the array goes into test, should be around 25%
x_val += filtered5[train1_length + test1_length:train1_length+valid1_length + test1_length]
y_val += filteredl5[train1_length + test1_length:train1_length+valid1_length + test1_length] #rest of the array goes into test, should be around 25%


x_train += filtered6[:train1_length] #stores 60% of array
y_train += filteredl6[:train1_length]
x_test += filtered6[train1_length:train1_length+test1_length] #stores 15% or array
y_test += filteredl6[train1_length:train1_length+test1_length] #rest of the array goes into test, should be around 25%
x_val += filtered6[train1_length + test1_length:train1_length+valid1_length + test1_length]
y_val += filteredl6[train1_length + test1_length:train1_length+valid1_length + test1_length] #rest of the array goes into test, should be around 25%


x_train += filtered7[:train1_length] #stores 60% of array
y_train += filteredl7[:train1_length]
x_test += filtered7[train1_length:train1_length+test1_length] #stores 15% or array
y_test += filteredl7[train1_length:train1_length+test1_length] #rest of the array goes into test, should be around 25%
x_val += filtered7[train1_length + test1_length:train1_length+valid1_length + test1_length]
y_val += filteredl7[train1_length + test1_length:train1_length+valid1_length + test1_length] #rest of the array goes into test, should be around 25%

x_train += filtered8[:train1_length] #stores 60% of array
y_train += filteredl8[:train1_length]
x_test += filtered8[train1_length:train1_length+test1_length] #stores 15% or array
y_test += filteredl8[train1_length:train1_length+test1_length] #rest of the array goes into test, should be around 25%
x_val += filtered8[train1_length + test1_length:train1_length+valid1_length + test1_length]
y_val += filteredl8[train1_length + test1_length:train1_length+valid1_length + test1_length] #rest of the array goes into test, should be around 25%

x_train += filtered9[:train1_length] #stores 60% of array
y_train += filteredl9[:train1_length]
x_test += filtered9[train1_length:train1_length+test1_length] #stores 15% or array
y_test += filteredl9[train1_length:train1_length+test1_length] #rest of the array goes into test, should be around 25%
x_val += filtered9[train1_length + test1_length:train1_length+valid1_length + test1_length]
y_val += filteredl9[train1_length + test1_length:train1_length+valid1_length + test1_length] #rest of the array goes into test, should be around 25%



x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
x_val = np.array(x_val)
y_val = np.array(y_val)



print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(x_val.shape)
print(y_val.shape)
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
history = model.fit(x_train, y_train,
                    validation_data = (x_val, y_val),
                    epochs=10,
                    batch_size=512)


# Report Results

print(history.history)
model.predict()