# Load the Drive helper and mount
from google.colab import drive

# This will prompt for authorization.
drive.mount('/content/drive')
!unzip -uq "/content/drive/MyDrive/brainimage" -d "/content/drive/MyDrive/brainimage"

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#%matplotlib inline
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import initializers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop

from keras.layers import Flatten, Dense, Dropout
from keras.models import Model
import random
import os

train_dir ='/content/drive/MyDrive/brainimage/clidical'
#test_dir ='/content/drive/MyDrive/brainimage/brainimage/test'
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    zoom_range=0.2,
    validation_split=0.4,
horizontal_flip=True,
fill_mode='nearest'
)
train_generator = train_datagen.flow_from_directory(train_dir,batch_size=20,target_size = (200,200),subset="training",class_mode = "binary")
val = train_datagen.flow_from_directory(train_dir,target_size = (200,200),subset="validation",class_mode='binary')

base_model = tf.keras.applications.InceptionV3(weights = "imagenet", include_top=False, input_shape = (200, 200, 3))

for layer in base_model.layers:
    layer.trainable = False

# Adding custom layers
x = base_model.output
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(2048, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

#model.summary()
from tensorflow.keras.optimizers import RMSprop
model.compile(loss='binary_crossentropy',
              optimizer='RMSprop',
              metrics=['accuracy']) 
history=model.fit(train_generator,epochs=15,validation_data=val,verbose=1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()