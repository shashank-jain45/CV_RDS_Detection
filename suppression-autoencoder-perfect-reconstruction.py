# Common
import os
import keras
import numpy as np
from glob import glob
import tensorflow as tf

# Images
import cv2 as cv
from keras.preprocessing.image import img_to_array

# Visualization
import matplotlib.pyplot as plt

# Model
from keras import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPool2D,Conv2DTranspose, Reshape, InputLayer

model_save_path = './working/bone_suppression_autoencoder.keras'
epoch_save_path = "./epochs/"
# Extract directory part
directory = os.path.dirname(model_save_path)
directory2 = os.path.dirname(epoch_save_path)

# Create directory if not present
os.makedirs(directory, exist_ok=True)
os.makedirs(directory2, exist_ok=True)  

print("Directory ready:", directory)

source_image_path = './archive/augmented/augmented/source/'
target_image_path = './archive/augmented/augmented/target/'

# Get Images
source_image_names = sorted(glob(source_image_path + "*.png"))
target_image_names = sorted(glob(target_image_path + "*.png"))

SIZE = 256

source_images = []
target_images = []

for img_path in source_image_names:
    img = cv.imread(img_path)
    img = cv.resize(img, (SIZE, SIZE), cv.INTER_CUBIC)
    img = img_to_array(img)
    img = img.astype('float')/255.
    source_images.append(img)

for img_path in target_image_names:
    img = cv.imread(img_path)
    img = cv.resize(img, (SIZE, SIZE), cv.INTER_CUBIC)
    img = img_to_array(img)
    img = img.astype('float')/255.
    target_images.append(img)



# Create directory if not present
source_images = np.array(source_images)
target_images = np.array(target_images)

image_shape = source_images.shape[-3:]

print("To check the number of images loaded: ")
print(len(source_images))
print(len(source_images))

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
plt.savefig(os.path.join(directory2,"image_grid.png"))

class EncoderLayerBlock(keras.layers.Layer):

    def __init__(self, filters):
        super(EncoderLayerBlock, self).__init__()
        self.filters = filters
        self.conv = Conv2D(filters, 3, padding='same', activation='relu')
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
        return {**base_config, "Filters": self.filters}


class ShowReconstruction(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        rand_id = np.random.randint(len(source_images))
        source_image = source_images[rand_id][np.newaxis, ...]
        reconstructed = self.model.predict(source_image)
        real_image = target_images[rand_id]

        plt.subplot(1, 3, 1)
        plt.imshow(source_image[0])
        plt.title("Source Image")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(reconstructed[0])
        plt.title("Produced Image")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(real_image)
        plt.title("Real Image")
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(directory2,"Reconstruction_Epoch_{}").format(epoch))


autoencoder = Sequential([

    # Encoder
    InputLayer(image_shape),  # image_shape is now (256, 256, 3)
    EncoderLayerBlock(32),
    EncoderLayerBlock(64),
    EncoderLayerBlock(128),

    # Latent Representation (Output is now 16x16x512)
    EncoderLayerBlock(512),

    # Decoder
    Conv2DTranspose(128, kernel_size=3, strides=2, padding='same', activation='relu'),

    # The rest remain 'same'
    Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', activation='relu'),
    Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', activation='relu'),
    Conv2DTranspose(3, kernel_size=3, strides=2, padding='same', activation='sigmoid'),

    Reshape(image_shape)  # This will correctly reshape to (256, 256, 3)
])
autoencoder.compile(
    loss='binary_crossentropy',
    optimizer= tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

autoencoder.fit(
    source_images, target_images,
    epochs=30,
    callbacks=[ShowReconstruction()]
)

# Save the entire model (architecture, weights, optimizer state, etc.)

autoencoder.save(model_save_path)
