#!/usr/bin/env python

# # **Imports**


import os 
import keras
import numpy as np 
from glob import glob
import tensorflow as tf


import cv2 as cv
from keras.preprocessing.image import img_to_array

import matplotlib.pyplot as plt

from keras import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPool2D,Conv2DTranspose, Reshape, InputLayer


# # **Image Data**


get_ipython().run_cell_magic('time', '', '\nsource_image_path = \'../input/xray-bone-shadow-supression/augmented/augmented/source/\'\ntarget_image_path = \'../input/xray-bone-shadow-supression/augmented/augmented/target/\'\n\n# Get Images\nsource_image_names = sorted(glob(source_image_path + "*.png"))\ntarget_image_names = sorted(glob(target_image_path + "*.png"))\n')




source_image_names[:10]




target_image_names[:10]



get_ipython().run_cell_magic('time', '', "SIZE = 256\n\nsource_images = []\ntarget_images = []\n\nfor img_path in source_image_names:\n    img = cv.imread(img_path)\n    img = cv.resize(img, (SIZE, SIZE), cv.INTER_CUBIC)\n    img = img_to_array(img)\n    img = img.astype('float')/255.\n    source_images.append(img)\n\nfor img_path in target_image_names:\n    img = cv.imread(img_path)\n    img = cv.resize(img, (SIZE, SIZE), cv.INTER_CUBIC)\n    img = img_to_array(img)\n    img = img.astype('float')/255.\n    target_images.append(img)\n\nsource_images = np.array(source_images)\ntarget_images = np.array(target_images)\n")




image_shape = source_images.shape[-3:]
image_shape



# # **Visualize**



plt.figure(figsize=(10,10))
for i in range(1,7):
    plt.subplot(3,2,i)
    if i%2!=0:
        rand_id = np.random.randint(len(source_images))
        plt.imshow(source_images[rand_id])
        plt.title('Source Image')
    elif i%2==0:
        plt.imshow(target_images[rand_id])
        plt.title('Target Image')
    plt.axis('off')
plt.tight_layout()
plt.show()


# # **AutoEncoder**




class EncoderLayerBlock(keras.layers.Layer):

    def __init__(self,filters):
        super(EncoderLayerBlock, self).__init__()
        self.filters = filters
        self.conv = Conv2D(filters,3,padding='same',activation='relu')
        self.bn = BatchNormalization()
        self.pool = MaxPool2D()

    def call(self, X):
        '''Takes the Input Image and pass it through the layers'''
        x = self.conv(X)
        x = self.bn(x)
        x = self.pool(x)

        return x

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "Filters":self.filters}



class ShowReconstruction(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        rand_id = np.random.randint(len(source_images))
        source_image = source_images[rand_id][np.newaxis,...]
        reconstructed = self.model.predict(source_image)
        real_image = target_images[rand_id]

        plt.subplot(1,3,1)
        plt.imshow(source_image[0])
        plt.title("Source Image")
        plt.axis('off')

        plt.subplot(1,3,2)
        plt.imshow(reconstructed[0])
        plt.title("Produced Image")
        plt.axis('off')

        plt.subplot(1,3,3)
        plt.imshow(real_image)
        plt.title("Real Image")
        plt.axis('off')

        plt.tight_layout()
        plt.savefig("Reconstruction_Epoch_{}".format(epoch))
        plt.show()




autoencoder = Sequential([

    # Encoder
    InputLayer(image_shape), # image_shape is now (256, 256, 3)
    EncoderLayerBlock(32),
    EncoderLayerBlock(64),
    EncoderLayerBlock(128),

    # Latent Representation (Output is now 16x16x512)
    EncoderLayerBlock(512),

    Conv2DTranspose(128, kernel_size=3, strides=2, padding='same', activation='relu'),


    Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', activation='relu'),
    Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', activation='relu'),
    Conv2DTranspose(3, kernel_size=3, strides=2, padding='same', activation='sigmoid'),

    Reshape(image_shape) 
])




autoencoder.compile(
    loss='binary_crossentropy',
    optimizer= tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)



autoencoder.fit(
    source_images, target_images,
    epochs=25,
    callbacks=[ShowReconstruction()]
)
model_save_path = '/kaggle/working/bone_suppression_autoencoder.keras'


autoencoder.save(model_save_path)


