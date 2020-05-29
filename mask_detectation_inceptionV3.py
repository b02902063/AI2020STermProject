import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
import os
import numpy as np
import matplotlib.pyplot as plt
from absl import app
from absl import flags
import shutil

FLAGS = flags.FLAGS
flags.DEFINE_bool ("test", False, "Testing mode")
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
def main(argv):
    checkpoint_path = "model/InceptionV3/cp.ckpt"          
    model = InceptionV3(weights=None , include_top=True, classes=1, classifier_activation=None, input_shape= (299, 299, 3), pooling='avg')
                
    if FLAGS.test:
        model.load_weights(checkpoint_path)  
        test_image_generator = ImageDataGenerator(rescale=1./255)
        test_data_gen = test_image_generator.flow_from_directory(batch_size=64,
                                                                directory="input/test",
                                                                 target_size=(299, 299),
                                                                 class_mode='binary') 

        files = test_data_gen.filenames                                                         
        old = 0
        all_count = 0
        fail_count = 0
        for d in test_data_gen:
            idx = (test_data_gen.batch_index - 1) * test_data_gen.batch_size
            if old > idx:
                break
            pred = model(d[0])
            fails = np.logical_or(np.logical_and(pred.numpy()[:, 0] < 0.5, d[1] > 0.5), np.logical_and(pred.numpy()[:, 0] > 0.5, d[1] < 0.5))
            for i, f in enumerate(fails):
                if f:
                    src = os.path.join("./input/test", files[idx + i])
                    dst = os.path.join("./fails/InceptionV3", files[idx + i].split('\\')[-1])
                    shutil.copyfile(src, dst)
            old = idx
            fail_count += np.sum(fails)
            all_count += 128
            
        print("Accuracy: {0}% ({1}/{2})".format(fail_count/all_count*100), fail_count, all_count)
        return
    else:
        model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])  
                  
        epochs = 15
        train_image_generator = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
        validation_image_generator = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
        train_data_gen  = train_image_generator.flow_from_directory(batch_size=32,
                                                                   directory="input/train",
                                                                   shuffle=True,
                                                                   target_size=(299, 299),
                                                                   class_mode='binary')
                                                                 
        val_data_gen = validation_image_generator.flow_from_directory(batch_size=32,
                                                                      directory="input/validation",
                                                                      target_size=(299, 299),
                                                                      class_mode='binary')      
        checkpoint_path = "model/InceptionV3/cp.ckpt"                                                                     
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)
                   
        history = model.fit_generator(
            train_data_gen,
            steps_per_epoch=74100 // 32,
            epochs=epochs,
            validation_data=val_data_gen,
            validation_steps=9370 // 32,
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
    
if __name__ == "__main__":
    app.run(main)