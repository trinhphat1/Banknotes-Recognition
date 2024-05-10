import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras import metrics
from keras.models import load_model
file = 'D:\\data1'
img_size = 32

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip= True,
                                   validation_split = 0.2)
test_datagen = ImageDataGenerator(rescale = 1./255,
                                  validation_split = 0.2)

train_datagen = train_datagen.flow_from_directory(file,
                                                  target_size = (img_size,img_size),
                                                  subset = 'training')
test_datagen = test_datagen.flow_from_directory(file, target_size = (img_size,img_size),
                                                subset = 'validation')

#bulid model
model = Sequential()

model.add(Conv2D(32, (3,3), padding = 'same', activation = 'relu',
                       input_shape = (32,32,3), kernel_initializer = 'he_uniform'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(32, (3,3), padding = 'same', activation = 'relu',
                       input_shape = (32,32,3), kernel_initializer = 'he_uniform'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0,5))

model.add(Conv2D(32, (3,3), padding = 'same', activation = 'relu',
                       input_shape = (32,32,3), kernel_initializer = 'he_uniform'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0,5))

model.add(Conv2D(32, (3,3), padding = 'same', activation = 'relu',
                       input_shape = (32,32,3), kernel_initializer = 'he_uniform'))
model.add(MaxPooling2D(pool_size = (2,2)))

model = Sequential()

model.add(Conv2D(32, (3,3), padding = 'same', activation = 'relu',
                       input_shape = (32,32,3), kernel_initializer = 'he_uniform'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Dense(512, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.summary()

model.compile(optimizer = Adam(), loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(train_datagen, epochs = 10, validation_data = test_datagen)
model.save('D:\\data123.keras')


