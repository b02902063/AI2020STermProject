import tensorflow as tf
from IPython.display import clear_output
import time


EPOCHS = 1000
BUFFER_SIZE = 1000
BATCH_SIZE = 32

chin_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(200, 200 ,3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1600),
    tf.keras.layers.Reshape([40, 40])
])

nose_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(200, 200 ,3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1600),
    tf.keras.layers.Reshape([40, 40])
])

mouth_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(200, 200 ,3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1600),
    tf.keras.layers.Reshape([40, 40])
])

chin_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
nose_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
mouth_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

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

@tf.function
def train_step(img, chin_label, nose_label, mouth_label):

    with tf.GradientTape(persistent=True) as tape:
        chin_predict = chin_model(img)
        nose_predict = nose_model(img)
        mouth_predict = mouth_model(img)
        
        chin_loss = loss_obj(chin_label, chin_predict)
        nose_loss = loss_obj(nose_label, nose_predict)
        mouth_loss = loss_obj(mouth_label, mouth_predict)
    
    # Calculate the gradients
    chin_gradients = tape.gradient(chin_loss, chin_model.trainable_variables)
    nose_gradients = tape.gradient(nose_loss, nose_model.trainable_variables)
    mouth_gradients = tape.gradient(mouth_loss, mouth_model.trainable_variables)
    
    # Apply the gradients to the optimizer
    chin_optimizer.apply_gradients(zip(chin_gradients, chin_model.trainable_variables))
    nose_optimizer.apply_gradients(zip(nose_gradients, nose_model.trainable_variables))
    mouth_optimizer.apply_gradients(zip(mouth_gradients, mouth_model.trainable_variables))
    
raw_dataset = tf.data.TFRecordDataset("helen.tfrecord")
dataset = raw_dataset.map(read_tfrecord)

dataset = dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

checkpoint_path = "./checkpoints"

ckpt = tf.train.Checkpoint(chin_model=chin_model,
                           nose_model=nose_model,
                           mouth_model=mouth_model,
                           chin_optimizer=chin_optimizer,
                           nose_optimizer=nose_optimizer,
                           mouth_optimizer=mouth_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')
for epoch in range(EPOCHS):
    start = time.time()

    n = 0
    for img, chin_label, nose_label, mouth_label in dataset:
        train_step(img, chin_label, nose_label, mouth_label)
        if n % 10 == 0:
            print ('.', end='')
        n+=1
    clear_output(wait=True)

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                         ckpt_save_path))

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                      time.time()-start))