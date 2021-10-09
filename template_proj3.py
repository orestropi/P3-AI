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
images = np.load("images.npy")
print(images.shape)
#images.resize((1,784))
print(images)
labels = np.load("labels.npy")
print(labels.shape)
array_labels = tf.keras.utils.to_categorical(labels, num_classes=None)

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

for label in labels:
    if label == 0:
        filtered0.append(True)
        filtered1.append(False)
        filtered2.append(False)
        filtered3.append(False)
        filtered4.append(False)
        filtered5.append(False)
        filtered6.append(False)
        filtered7.append(False)
        filtered8.append(False)
        filtered9.append(False)
    elif label == 1:
        filtered0.append(False)
        filtered1.append(True)
        filtered2.append(False)
        filtered3.append(False)
        filtered4.append(False)
        filtered5.append(False)
        filtered6.append(False)
        filtered7.append(False)
        filtered8.append(False)
        filtered9.append(False)
    elif label == 2:
        filtered0.append(False)
        filtered1.append(False)
        filtered2.append(True)
        filtered3.append(False)
        filtered4.append(False)
        filtered5.append(False)
        filtered6.append(False)
        filtered7.append(False)
        filtered8.append(False)
        filtered9.append(False)
    elif label == 3:
        filtered0.append(False)
        filtered1.append(False)
        filtered2.append(False)
        filtered3.append(True)
        filtered4.append(False)
        filtered5.append(False)
        filtered6.append(False)
        filtered7.append(False)
        filtered8.append(False)
        filtered9.append(False)
    elif label == 4:
        filtered0.append(False)
        filtered1.append(False)
        filtered2.append(False)
        filtered3.append(False)
        filtered4.append(True)
        filtered5.append(False)
        filtered6.append(False)
        filtered7.append(False)
        filtered8.append(False)
        filtered9.append(False)
    elif label == 5:
        filtered0.append(False)
        filtered1.append(False)
        filtered2.append(False)
        filtered3.append(False)
        filtered4.append(False)
        filtered5.append(True)
        filtered6.append(False)
        filtered7.append(False)
        filtered8.append(False)
        filtered9.append(False)
    elif label == 6:
        filtered0.append(False)
        filtered1.append(False)
        filtered2.append(False)
        filtered3.append(False)
        filtered4.append(False)
        filtered5.append(False)
        filtered6.append(True)
        filtered7.append(False)
        filtered8.append(False)
        filtered9.append(False)
    elif label == 7:
        filtered0.append(False)
        filtered1.append(False)
        filtered2.append(False)
        filtered3.append(False)
        filtered4.append(False)
        filtered5.append(False)
        filtered6.append(False)
        filtered7.append(True)
        filtered8.append(False)
        filtered9.append(False)
    elif label == 8:
        filtered0.append(False)
        filtered1.append(False)
        filtered2.append(False)
        filtered3.append(False)
        filtered4.append(False)
        filtered5.append(False)
        filtered6.append(False)
        filtered7.append(False)
        filtered8.append(True)
        filtered9.append(False)
    else:
        filtered0.append(False)
        filtered1.append(False)
        filtered2.append(False)
        filtered3.append(False)
        filtered4.append(False)
        filtered5.append(False)
        filtered6.append(False)
        filtered7.append(False)
        filtered8.append(False)
        filtered9.append(True)

filtered1 = np.array(filtered1)
filtered1 = filtered1.reshape((6500, 1)) #match indexes, not sure if needed
#print(len(filtered1))
#print(filtered1.shape)
#print(images.shape)
#print(labels.shape)
one = images[filtered1] #error is here ***** something with array shapes when trying to filter
rng = np.random.default_rng()
one = rng.shuffle(one) #shuffle array
train1_length = round(len(one)*.6) #gets 60% of array
valid1_length = round(len(one)*.15) #gets 15% of array
train1 = one[:train1_length] #stores 60% of array
np.delete(one, np.arange(train1_length)) #deletes 60% of the array
valid1 = one[:valid1_length] #stores 15% or array
np.delete(train1, np.arrange(valid1_length)) #deletes 15% of array
test1 = one #rest of the array goes into test, should be around 25%

two = images[filtered2]
three = images[filtered3]
four = images[filtered4]
five = images[filtered5]
six = images[filtered6]
seven = images[filtered7]
eight = images[filtered8]
nine = images[filtered9]
zero = images[filtered0]



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