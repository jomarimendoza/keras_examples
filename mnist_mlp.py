"""
deep neural network with fully-connected layers using MNIST dataset
"""

"""
Simple neural network using Multi-layer Perceptron
accuracy: 98.1%
"""

import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras.optimizers import RMSprop
from keras.utils import to_categorical

BATCH_SIZE = 16
NUM_CLASSES = 10
EPOCHS = 30

def mnist_model(layer_input):
    y = Dense(512,activation='relu')(layer_input)
    y = Dropout(0.2)(y)
    y = Dense(512,activation='relu')(y)
    y = Dropout(0.2)(y)
    y = Dense(NUM_CLASSES,activation='softmax')(y)
    return y

(X_train, y_train), (X_test,y_test) = mnist.load_data()

num_px = 28*28

X_train = X_train.reshape(-1,num_px).astype('float32') / 255
X_test = X_test.reshape(-1,num_px).astype('float32') / 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# one-hat vector
y_train = to_categorical(y_train,NUM_CLASSES)
y_test = to_categorical(y_test,NUM_CLASSES)

input_layer = Input((num_px,))
output = mnist_model(input_layer)
mlp = Model(input_layer,output)

mlp.summary()

rmsprop = RMSprop()

mlp.compile(loss='categorical_crossentropy',
            optimizer=rmsprop,
            metrics=['accuracy'])
history = mlp.fit(X_train,y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    verbose=1,
                    validation_data=(X_test,y_test))

score = mlp.evaluate(X_test,y_test,verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
