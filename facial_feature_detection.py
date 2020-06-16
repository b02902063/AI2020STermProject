import tensorflow as tf
from IPython.display import clear_output
import time
import cv2
import numpy as np
from models.FCN import *
import os
import shutil
import tensorflow_addons as tfa
import sys


overlay_colors = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0], [0, 255, 255], [255, 0, 255], [192, 192, 192], [128, 0, 0], [0, 128, 0], [255, 255, 255], [128, 0, 128], [0, 128, 128], [0, 0, 128]]
num_class = 5
n_m_label = [2, 3]

directory = "helenstar_5_all"
def my_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
"""
try:
    shutil.rmtree("./{0}".format(directory))
except:
    pass
"""

my_mkdir("./{0}".format(directory))
my_mkdir("./{0}/mix_model".format(directory))
my_mkdir("./{0}/test_proce".format(directory))
my_mkdir("./{0}/train_proce".format(directory))
my_mkdir("./{0}/test_proce/pred".format(directory))
my_mkdir("./{0}/train_proce/pred".format(directory))
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
EPOCHS = 100000
BUFFER_SIZE = 1000
BATCH_SIZE = 1

model = FCN8s((224,224,3), num_class)
bgr_mean = np.array([0.406, 0.456, 0.485])
bgr_std = np.array([0.225, 0.224, 0.229])

def write_processdure(img, label=None, count=0, path="./{0}/test_proce".format(directory), nose_mouth=False, find_suspicious=False):

    predict = model(img, training=False)
    if type(predict) == tuple:
        predict = predict[0]
    predict = predict.numpy()[0]
    predict = np.argmax(predict, axis=-1)
    
    img = img.numpy()[0]
    img = (img * bgr_std + bgr_mean) * 255
    img = img.astype(np.uint8)
    if label is not None:
        label = label.numpy()[0]
    def draw_overlay(img, indices, color):
        rec_result = img.copy()
        overlay = img.copy()
        overlay[indices] = color
        return cv2.addWeighted(overlay, 0.5, rec_result, 0.5, 0) 
        
    predict_img = img.copy()
    truth_img = img.copy()
    
    pred_n_m_count = 0
    face_count = (img.shape[0] * img.shape[1]) - len(np.where(predict==0)[0])
    if nose_mouth:
        for i in [1, 2, 3 ,4]:
            n_m = np.where(predict==i)
            predict_img = draw_overlay(predict_img, n_m, overlay_colors[i])
            if i in n_m_label:
                pred_n_m_count += len(n_m[0])
            if label is not None:
                truth_img = draw_overlay(truth_img, np.where(label==i), [0, 0, 255])
    else:
        for i in range(1, num_class):
            predict_img = draw_overlay(predict_img, np.where(predict==i), overlay_colors[i])
            if label is not None:
                truth_img = draw_overlay(truth_img, np.where(label==i), overlay_colors[i])
    if label is not None:
        seperate_line = np.zeros(shape=(img.shape[0], 5, img.shape[2]), dtype=np.uint8)
        seperate_line[..., 1] = 76
        seperate_line[..., 2] = 153

        rrt = np.concatenate([predict_img, seperate_line, truth_img], axis=1)
    else:
        rrt = predict_img
      
    if find_suspicious:
        if face_count == 0:
            cv2.imwrite("./{0}/pred/{1}_{2}_{3}.jpg".format(path, count, face_count, pred_n_m_count), rrt)
        elif pred_n_m_count/face_count > 0.04 and pred_n_m_count > 1000:
            cv2.imwrite("./{0}/suspicious/{1}_{2}_{3}.jpg".format(path, count, round(pred_n_m_count/face_count, 4), pred_n_m_count), rrt)
        else:
            cv2.imwrite("./{0}/pred/{1}_{2}_{3}.jpg".format(path, count, pred_n_m_count/face_count, pred_n_m_count), rrt)
    else:
        cv2.imwrite("./{0}/pred/{1}.jpg".format(path, count), rrt)
    return pred_n_m_count

def read_tfrecord(serialized_example):
    # Create a description of the features.
    feature_description = {
        'img': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'label': tf.io.FixedLenFeature([], tf.string, default_value='')
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    img = tf.io.parse_tensor(example['img'], out_type=tf.int32)
    label = tf.io.parse_tensor(example['label'], out_type=tf.int32)
    img = tf.cast(img, tf.float32)
    #label = tf.cast(label, tf.float32)
    #img /= 127.5
    #img -= 1.0
    img = (img / 255. - bgr_mean) / bgr_std
    #label = tf.concat([tf.ones_like(label)[:, :112], tf.ones_like(label)[:, 112:]*2], axis=1)
    #label = tf.ones_like(label)
    img.set_shape((224, 224, 3))
    label.set_shape((224, 224))
    return img, label

def read_masked_tfrecord(serialized_example):
    # Create a description of the features.
    feature_description = {
        'img': tf.io.FixedLenFeature([], tf.string, default_value=''),
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    img = tf.io.parse_tensor(example['img'], out_type=tf.int32)
    img = tf.cast(img, tf.float32)
    img = (img / 255. - bgr_mean) / bgr_std
    img.set_shape((224, 224, 3))
    return img

@tf.function
def train_step(img, label):

    with tf.GradientTape(persistent=True) as tape:
        predict = model(img, training=True)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(label, predict)
    
    # Calculate the gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Apply the gradients to the optimizer
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
raw_dataset = tf.data.TFRecordDataset("helenstar_5_categorical.tfrecord")
#raw_masked_dataset = tf.data.TFRecordDataset("masked.tfrecord")

train_dataset = raw_dataset.take(1699)
train_dataset = train_dataset.map(read_tfrecord)
validation_dataset = raw_dataset.skip(1699)
validation_dataset = validation_dataset.map(read_tfrecord).batch(1)

dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
#raw_masked_dataset = 
checkpoint_path = "./model/nose_mouth_eye/nme.ckpt"  #FCN8s_{epoch:04d}-{val_accuracy:.2f}.ckpt

"""
ckpt = tf.train.Checkpoint(model=model,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')
"""    

if len(sys.argv) > 1:       
    try:
        loaded_checkpoint_path = tf.train.latest_checkpoint("./model/nose_mouth_eye/")
        model.load_weights(loaded_checkpoint_path)
        print("Restore file successfully")
    except:
        print("Didn't find checkpoint. Exit.")
        exit()
    if sys.argv[1] == "mask":
        mask_dataset = tf.data.TFRecordDataset("masked224.tfrecord")
        mask_dataset = mask_dataset.map(read_masked_tfrecord).batch(1)
        count = 0
        nm_count = []
        for img in mask_dataset:
            nm_count.append(write_processdure(img, None, count, "./{0}/masked_proce".format(directory), nose_mouth=True, find_suspicious=True))
            count += 1
        print(sum(nm_count)/count, max(nm_count), min(nm_count))
    elif sys.argv[1] == "face":
        face_dataset = tf.data.TFRecordDataset("face224.tfrecord")
        face_dataset = face_dataset.map(read_masked_tfrecord).batch(1)
        count = 0
        nm_count = []
        for img in face_dataset:
            nm_count.append(write_processdure(img, None, count, "./{0}/face_proce".format(directory), nose_mouth=True))
            count += 1
        print(sum(nm_count)/count, max(nm_count), min(nm_count))
    else:
        train_dataset = train_dataset.batch(1)
        test_dataset = raw_dataset.skip(1699).map(read_tfrecord).batch(1)
        count = 0
        for img, label in test_dataset:
            write_processdure(img, label, count)
            count += 1
        count = 0
        for img, label in train_dataset:
            write_processdure(img, label, count, "./{0}/train_proce".format(directory))
            count += 1
    exit()
try:    
    loaded_checkpoint_path = tf.train.latest_checkpoint("./model/nose_mouth_eye/")
    model.load_weights(loaded_checkpoint_path)
    print("Restore file successfully")
except:
    print("Didn't find checkpoint. Initialize a new model.")
model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])   
           
callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True, monitor="val_accuracy", save_best_only=True)

def generator(dataset):
    while True:
        for i, l in dataset:
            yield i, l

model.fit(dataset, epochs=1000, callbacks=[callback], validation_data=validation_dataset)              
