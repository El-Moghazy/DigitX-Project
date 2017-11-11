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
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load data
# test = pd.read_csv('test.csv')
# training = pd.read_csv('train.csv')
# X_test2 = test
# train = ["pixel" + str(x) for x in range(784)]
# X_train = training[train]
# y_train = training['label']
# X_test2 = X_test2.as_matrix()
# X_train = X_train.as_matrix()
# y_train = y_train.as_matrix()

num_classes = 10

(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test2 = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)

y_test = np_utils.to_categorical(y_test)


# define the larger model
def model():
    # create model
    model = Sequential()
    model.add(Conv2D(20, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(50, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# build the model
model = model()
# Fit the model
model.fit(X_train, y_train, epochs=10, batch_size=200)
# Final evaluation of the model
# scores = model.evaluate(X_test, y_test, verbose=0)
# print("Large CNN Error: %.2f%%" % (100-scores[1]*100))
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

result = model.evaluate(X_test, y_test, verbose=10)
# result = numpy.argmax(result, axis=1)
# result = pd.Series(result, name="Label")
# submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), result], axis=1)
# submission.to_csv("C:/Users/ElMoghazy/Desktop/f.csv", index=False)
