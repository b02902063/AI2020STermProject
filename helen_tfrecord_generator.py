import cv2
import tensorflow as tf
import numpy as np
import os


all_img = []
all_label = []


for file in os.listdir("./helenstar_release/train"):
    if "image" not in file:
        continue
        
    image_file = "./helenstar_release/train/" + file
    label_file = "./helenstar_release/train/" + file.split("image")[0] + "label.png"
    img = cv2.imread(image_file)
    label = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)
    
    img = cv2.resize(img, (224, 224))
    label = cv2.resize(label, (224, 224), interpolation=cv2.INTER_NEAREST)
    
    label[np.logical_and.reduce((label != 6, label != 7, label != 8, label != 9, label != 0, label != 4, label != 5))] = 1
    label[label == 6] = 2 # nose
    label[label == 7] = 3 # mouth
    label[label == 8] = 3
    label[label == 9] = 3
    label[label == 4] = 4 # eye
    label[label == 5] = 4
    all_img.append(img)
    all_label.append(label)
    
dataset = tf.data.Dataset.from_tensor_slices((all_img, all_label))

def serialize_example(img, label):

    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() 
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    """
    Creates a tf.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    feature = {
        'img': _bytes_feature(tf.io.serialize_tensor(img)),
        'label': _bytes_feature(tf.io.serialize_tensor(label))
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def tf_serialize_example(img, label):
    tf_string = tf.py_function(
        serialize_example,
        (img, label),  # pass these args to the above function.
        tf.string)      # the return type is `tf.string`.
    return tf.reshape(tf_string, ()) # The result is a scalar
serialized_features_dataset = dataset.map(tf_serialize_example)
writer = tf.data.experimental.TFRecordWriter("helenstar_5_categorical.tfrecord")
writer.write(serialized_features_dataset)
