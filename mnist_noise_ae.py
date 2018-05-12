"""
Denoising autoencoder on MNIST dataset
"""
"""
Input: Noisy image
Output: Clean Image
"""

import keras
from keras import backend as K
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.datasets import mnist

import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# for randomization
np.random.seed(1231 )

# Parameters
BATCH_SIZE = 100
EPOCHS = 50

def encoder(input_layer, latent_dim):
    y = Conv2D(filters=32, kernel_size=3, strides=2, activation='relu', padding='same')(input_layer)
    before_latent = Conv2D(filters=64, kernel_size=3, strides=2, activation='relu', padding='same')(y)
    map_shape = K.int_shape(before_latent)

    y = Flatten()(before_latent)
    latent = Dense(latent_dim, name='latent_vector')(y)

    # returns output and shape of output in decoder
    return latent, map_shape

def decoder(input_layer, map_shape):
    y = Dense(map_shape[1] * map_shape[2] * map_shape[3])(input_layer)
    y = Reshape(map_shape[1:])(y)
    y = Conv2DTranspose(filters=64, kernel_size=3, strides=2,activation='relu',padding='same')(y)
    y = Conv2DTranspose(filters=32, kernel_size=3, strides=2,activation='relu',padding='same')(y)
    y = Conv2DTranspose(filters=1, kernel_size=3, padding='same')(y)

    output = Activation('sigmoid', name='decoder_output')(y)
    return output

(X_train,_), (X_test,_) = mnist.load_data()

image_size = (X_train.shape[1], X_train.shape[2])
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test , axis=-1)
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Corrupt MNIST data (normal dist noise)
noise = np.random.normal(loc=0.5, scale=0.5, size=X_train.shape)
X_train_noisy = X_train + noise
noise = np.random.normal(loc=0.5, scale=0.5, size=X_test.shape)
X_test_noisy = X_test + noise
# Determine values greater than 1 (set to 1)
X_train_noisy = np.clip(X_train_noisy, 0., 1.)
X_test_noisy = np.clip(X_test_noisy, 0., 1.)

# Build ENCODER
LATENT_DIM = 16
input = Input(shape=(*image_size,1), name='encoder_input')
latent,map_shape = encoder(input,LATENT_DIM)

enc = Model(input,latent,name='encoder')
enc.summary()

# Build DECODER
latent_input = Input(shape=(LATENT_DIM,), name='decoder_input')
output = decoder(latent_input,map_shape)

dec = Model(latent_input,output,name='decoder')
dec.summary()

# Build AUTO-ENCODER
# Create a model from 2 instantiated models
output_AE = dec(enc(input))
autoencoder = Model(input,output_AE,name='autoencoder')
autoencoder.summary()

autoencoder.compile(loss='mse',optimizer='adam',metrics=['accuracy'])

# Training the data
autoencoder.fit(X_train_noisy,
                X_train,
                validation_data=(X_test_noisy,X_test),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE)

# Predict the Autoencoder output from corrupted test images
X_decoded = autoencoder.predict(X_test_noisy)

# Display the 1st 8 corrupted and denoised images
rows, cols = 10, 30
num = rows * cols
imgs = np.concatenate([X_test[:num], X_test_noisy[:num], X_decoded[:num]])
imgs = imgs.reshape((rows * 3, cols, *image_size))
imgs = np.vstack(np.split(imgs, rows, axis=1))
imgs = imgs.reshape((rows * 3, -1, *image_size))
imgs = np.vstack([np.hstack(i) for i in imgs])
imgs = (imgs * 255).astype(np.uint8)
plt.figure()
plt.axis('off')
plt.title('Original images: top rows, '
          'Corrupted Input: middle rows, '
          'Denoised Input:  third rows')
plt.imshow(imgs, interpolation='none', cmap='gray')
misc.imsave('corrupted_and_denoised.png',imgs)
plt.show()

scores = autoencoder.evaluate(X_train_noisy,X_train,verbose=1)

print('\nloss: ', scores[0])
print('acc: ', scores[1])

# Save model and weights
save_dir = os.path.join(os.getcwd(),'saved_models')
model_name = 'keras_mnist_autoencoder_model.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
autoencoder.save(model_path)
print('Saved trained model at %s ' % model_path)
