import tensorflow as tf


BUFFER_SIZE = 1000
BATCH_SIZE = 32

def read_tfrecord(serialized_example):
    # Create a description of the features.
    feature_description = {
        'img': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'chin_label': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'nose_label': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'mouth_label': tf.io.FixedLenFeature([], tf.string, default_value=''),
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    img = tf.io.parse_tensor(example['img'], out_type=tf.int32)
    chin_label = tf.io.parse_tensor(example['chin_label'], out_type=tf.float32)
    nose_label = tf.io.parse_tensor(example['nose_label'], out_type=tf.float32)
    mouth_label = tf.io.parse_tensor(example['mouth_label'], out_type=tf.float32)
    return img, chin_label, nose_label, mouth_label

raw_dataset = tf.data.TFRecordDataset("helen.tfrecord")
dataset = raw_dataset.map(read_tfrecord)

dataset = dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
for img, chin_label, nose_label, mouth_label in dataset:
    #Training
    pass
    
    
    