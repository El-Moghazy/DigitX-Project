# Larger CNN for the MNIST Dataset
# Modified from https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/ and from https://machinelearningmastery.com/save-load-keras-deep-learning-models/
import numpy
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import np_utils

K.set_image_dim_ordering('th')

seed = 0
numpy.random.seed(seed)

num_classes = 10

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshaping the samples according to theano convention samples,channels,rows,columns as we are using image_dim_ordering('th') if you are using 
# image_dim_ordering('tf') you will need to change the order to samples,rows,column,channels and here we have only one channel because we are 
# using the image in grayscale not RGB.
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# normalize the inputs by dividing / 255 because the maximum value that can be stored in the pixel is 255 so it will give the range from 0 to 1
X_train = X_train / 255
X_test2 = X_test / 255

# making the output in the form of one vs all (aka one hot encoding) which means that we will have 10 calsses from 0 to 10 one class or each number from 0 to 9
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Our model will be a sequentiol model consisting of two convolutional layer and two fully connected layers : Conv -> MaxPool -> Dropout -> Conv -> MaxPool -> Dropout -> fullyConnected -> Fully connected
model = Sequential()
# We have to state the dimentions of the image intering the conv layer in the first layer only then it is handled by Keras
model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))

# Using maxpooling layer after convolutional layer is important to reduce the risk of overfitting as 
# it reduces the image size which removes some details and also reduces the computational cost
model.add(MaxPooling2D(pool_size=(2, 2)))

# we use drop out as a method to regulize the model and prevent the CNN from overfitting
model.add(Dropout(0.2))

model.add(Conv2D(70, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
#Early Stopping
#keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Fit the model
model.fit(X_train, y_train, epochs=15, batch_size=200)


# Evaluating the model on unseen data to be able to do the error analysis and know if we have bias or variance problems
scores = model.evaluate(X_test, y_test, verbose = 10)

# Saving the model for future use
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Model Saved successfully")
