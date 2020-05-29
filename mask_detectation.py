import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt
epochs = 15
train_image_generator = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
validation_image_generator = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
train_data_gen  = train_image_generator.flow_from_directory(batch_size=128,
                                                           directory="input/train",
                                                           shuffle=True,
                                                           target_size=(150, 150),
                                                           class_mode='binary')
                                                         
val_data_gen = validation_image_generator.flow_from_directory(batch_size=128,
                                                              directory="input/validation",
                                                              target_size=(150, 150),
                                                              class_mode='binary')      
                                                              
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(150, 150 ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
              
checkpoint_path = "model/cp.ckpt"
        
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
           
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=74100 // 128,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=9370 // 128,
    callbacks=[cp_callback]
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()