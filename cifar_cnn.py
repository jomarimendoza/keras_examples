"""
deep CNN using CIFAR10 small images dataset
"""
"""
Using Data Augmentation to increase number of training data
accuracy: 75.76%
"""

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense,Dropout,Activation,Flatten,Input
from keras.layers import Conv2D,MaxPooling2D
from keras.utils import to_categorical
import os

BATCH_SIZE = 64
NUM_CLASSES = 10
EPOCHS = 100
AUGMENT = True
NUM_PREDICTIONS = 20


def cifar_model(layer_input):
    y = Conv2D(32,kernel_size=(3,3),padding='same')(layer_input)
    y = Activation('relu')(y)
    y = Conv2D(32,kernel_size=(3,3))(y)
    y = Activation('relu')(y)
    y = MaxPooling2D(pool_size=(2,2))(y)
    y = Dropout(0.25)(y)

    y = Conv2D(64,kernel_size=(3,3),padding='same')(y)
    y = Activation('relu')(y)
    y = Conv2D(64,kernel_size=(3,3))(y)
    y = Activation('relu')(y)
    y = MaxPooling2D(pool_size=(2,2))(y)
    y = Dropout(0.25)(y)

    y = Flatten()(y)
    y = Dense(512)(y)
    y = Activation('relu')(y)
    y = Dropout(0.5)(y)
    output = Dense(NUM_CLASSES,activation='softmax')(y)
    return output


save_dir = os.path.join(os.getcwd(),'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# load data
(X_train,y_train),(X_test,y_test) = cifar10.load_data()

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

rmsprop = keras.optimizers.rmsprop(lr=0.0001,decay=1e-6)

input = Input((32,32,3))

output = cifar_model(input)

cnn = Model(input,output)

cnn.compile(loss='categorical_crossentropy',
            optimizer=rmsprop,
            metrics=['accuracy'])

if not AUGMENT:
    print('Not using data augmentation.')
    cnn.fit(X_train,y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_test,y_test),
            shuffle=True)

else:
    print('Using real-time data augmentation')
    # pre-processing
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen.fit(X_train)

    num_data = X_train.shape[0]
    cnn.fit_generator(datagen.flow(X_train,y_train,
                                    batch_size=BATCH_SIZE),
                                    steps_per_epoch=num_data//BATCH_SIZE,
                                    epochs=EPOCHS,
                                    validation_data=(X_test,y_test),
                                    workers=4)

    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    cnn.save(model_path)
    print('Saved trained model at %s ' % model_path)

    # Score
    scores = cnn.evaluate(X_test,y_test,verbose=1)

    print('\nloss: ', scores[0])
    print('acc: ', scores[1])
