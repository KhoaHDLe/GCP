##############################################################
# Importing Libraries
##############################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import cv2
import os

from IPython.display import Image
from keras.preprocessing import image
from tqdm import tqdm

from keras import optimizers
from keras import layers, models
from keras.applications.imagenet_utils import preprocess_input
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

##############################################################
# Setting up Parameters
##############################################################

batch_size = 64
target_size = (32, 32)
class_mode = 'binary'
epochs = 100
input_shape = (32, 32, 3)
num_classes = 2
train_dir = "C:/Users/Khoa/Desktop/Coding/0.0-Project-Imaging/Kaggle/Aerial_Cactus_Identification/Data/train/"
validation_split = 0.8
color_mode = 'rgb'
x_col = 'id'
y_col = 'has_cactus'
dropout_dense_layer = 0.5

###########################################################################################################
# Splitting training and validation into approximately 80% to 20% split
###########################################################################################################

df = pd.read_csv("C:/Users/Khoa/Desktop/Coding/0.0-Project-Imaging/Kaggle/Aerial_Cactus_Identification/train.csv")

df.has_cactus = df.has_cactus.astype(str)
msk = np.random.rand(len(df)) < validation_split
train = df[msk]
validation = df[~msk]

##############################################################
# Data Augmentation and Feed
##############################################################

data_generator = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    vertical_flip=True)

train_generator = data_generator.flow_from_dataframe(
    color_mode=color_mode,
    dataframe=train,
    directory=train_dir,
    x_col=x_col,
    y_col=y_col,
    batch_size=batch_size,
    shuffle=True,
    class_mode=class_mode,
    target_size=target_size)

validation_generator = data_generator.flow_from_dataframe(
    color_mode=color_mode,
    dataframe=validation,
    directory=train_dir,
    x_col=x_col,
    y_col=y_col,
    batch_size=batch_size,
    shuffle=True,
    class_mode=class_mode,
    target_size=target_size)

##############################################################
# Defining model structure
##############################################################

from keras.models import Sequential
from keras.layers import InputLayer, Input
from keras.layers import Conv2D, Dense, Flatten, Dropout, Activation
from keras.layers import BatchNormalization, Reshape, MaxPooling2D, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(GlobalAveragePooling2D())

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(dropout_dense_layer))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy",
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

##############################################################
# Fitting the CNN to the Data
##############################################################

STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size  # 15000/50 = 300
STEP_SIZE_VALID = validation_generator.n // validation_generator.batch_size  # 2500/50 = 50

earlystopper = EarlyStopping(monitor='val_loss', patience=12, verbose=1, mode='auto', restore_best_weights=True)
reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='auto', min_delta=0.0001, cooldown=1)

history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=validation_generator,
    validation_steps=STEP_SIZE_VALID,
    epochs=epochs,
    callbacks=[earlystopper, reducelr])

model.save("CNN_model_v_0.1.h5")
print("Saved model to disk")


##############################################################
# Plotting accuracy metrics
##############################################################

plt.figure(figsize=(15, 5))

plt.subplot(141)
plt.plot(history.history['loss'], label='training')
plt.plot(history.history['val_loss'], label='validation')
plt.xlabel('# Epochs')
plt.legend()
plt.ylabel("Loss - Binary Cross Entropy")
plt.title('Loss Evolution')
plt.show()

plt.figure(figsize=(15, 5))
plt.subplot(142)
plt.plot(history.history['loss'], label='training')
plt.plot(history.history['val_loss'], label='validation')
plt.ylim(0, 0.1)
plt.xlabel('# Epochs')
plt.legend()
plt.ylabel("Loss - Binary Cross Entropy")
plt.title('Zoom Near Zero - Loss Evolution')
plt.show()

plt.figure(figsize=(15, 5))

plt.subplot(143)
plt.plot(history.history['acc'], label='training')
plt.plot(history.history['val_acc'], label='validation')
plt.xlabel('# Epochs')
plt.ylabel("Accuracy")
plt.legend()
plt.title('Accuracy Evolution')
plt.show()

plt.figure(figsize=(15, 5))
plt.subplot(144)
plt.plot(history.history['acc'], label='training')
plt.plot(history.history['val_acc'], label='validation')
plt.ylim(0.98, 1)
plt.xlabel('# Epochs')
plt.ylabel("Accuracy")
plt.legend()
plt.title('Zoom Near One - Accuracy Evolution')
plt.show()
