import tensorflow as tf
import sys

def FCN8s(input_shape, num_class):

    inp = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.ZeroPadding2D(100)(inp)
    x = tf.keras.layers.Conv2D(64, 3, padding='valid', activation='relu')(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    pool1 = tf.keras.layers.MaxPooling2D(padding="same")(x)

    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(pool1)
    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    pool2 = tf.keras.layers.MaxPooling2D(padding="same")(x)
    
    x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(pool2)
    x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    pool3 = tf.keras.layers.MaxPooling2D(padding="same")(x)
    
    x = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(pool3)
    x = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    pool4 = tf.keras.layers.MaxPooling2D(padding="same")(x)
    
    x = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(pool4)
    x = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    pool5 = tf.keras.layers.MaxPooling2D(padding="same")(x)
    
    x = tf.keras.layers.Conv2D(4096, 7, padding='valid', activation='relu')(pool5)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.Conv2D(4096, 1, padding='valid', activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    score_fr = tf.keras.layers.Conv2D(num_class, 1)(x)
    
    upscore2 = tf.keras.layers.Conv2DTranspose(num_class, 4, strides=2, padding='valid', use_bias=False)(score_fr)
    
    h =  tf.keras.layers.Conv2D(num_class, 1)(pool4 * 0.01)
    score_pool4c = h[:, 5:5+upscore2.shape[1], 5:5+upscore2.shape[2], :]
    #print(upscore2, score_pool4c, h)
    h = upscore2 + score_pool4c
    upscore_pool4  = tf.keras.layers.Conv2DTranspose(num_class, 4, strides=2, padding='valid', use_bias=False)(h)
    
    h = tf.keras.layers.Conv2D(num_class, 1)(pool3 * 0.0001)
    score_pool3c = h[:, 9:9+upscore_pool4.shape[1], 9:9+upscore_pool4.shape[2], :]
    
    h = upscore_pool4 + score_pool3c
    h = tf.keras.layers.Conv2DTranspose(num_class, 16, strides=8, padding='valid', use_bias=False)(h)
    h = h[:, 31:31+input_shape[0], 31:31+input_shape[1], :]
    
    #h = tf.nn.softmax(h, axis=-1)
    return tf.keras.models.Model(inputs=inp, outputs=h)
    

def Resnet50_FCN(input_shape, num_class):

    def attention(x):
    
        def flatten_hw(x):
            return tf.reshape(x, shape=(-1, x.shape[1] * x.shape[2], x.shape[3]))
    
        batch_size, height, width, channels = x.get_shape().as_list()
        f = tf.keras.layers.Conv2D(channels//8, 1, strides=1, padding='valid', activation='relu')(x)
        f = tf.keras.layers.MaxPooling2D(padding="same")(f)
        
        g = tf.keras.layers.Conv2D(channels//8, 1, strides=1, padding='valid', activation='relu')(x)
        
        h = tf.keras.layers.Conv2D(channels//2, 1, strides=1, padding='valid', activation='relu')(x)
        h = tf.keras.layers.MaxPooling2D(padding="same")(h)

        s = tf.linalg.matmul(flatten_hw(g), flatten_hw(f), transpose_b=True)
        beta = tf.nn.softmax(s)
        o = tf.linalg.matmul(beta, flatten_hw(h))
        
        gamma = tf.Variable(0.0, trainable=True, dtype=tf.float32)
        o = tf.reshape(o, shape=[-1, height, width, channels // 2])
        o = tf.keras.layers.Conv2D(channels, 1, strides=1, padding='valid', activation='relu')(o)

        x = gamma * o + x
        return x

    def conv_shortcut_block(kernel_size, filters, strides=(2, 2)):
        def helper(input_tensor):

            x = tf.keras.layers.Conv2D(filters[0], (1, 1), strides=strides, kernel_regularizer=tf.keras.regularizers.l2(0))(input_tensor)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)

            x = tf.keras.layers.Conv2D(filters[1], kernel_size, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0))(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)

            x = tf.keras.layers.Conv2D(filters[2], (1, 1), kernel_regularizer=tf.keras.regularizers.l2(0))(x)
            x = tf.keras.layers.BatchNormalization()(x)

            shortcut = tf.keras.layers.Conv2D(filters[2], (1, 1), strides=strides, kernel_regularizer=tf.keras.regularizers.l2(0))(input_tensor)
            shortcut = tf.keras.layers.BatchNormalization()(shortcut)

            x = tf.keras.layers.Add()([x, shortcut])
            x = tf.keras.layers.Activation('relu')(x)
            return x
        return helper
        
    def block(kernel_size, filters):
        def helper(input_tensor):

            x = tf.keras.layers.Conv2D(filters[0], (1, 1), kernel_regularizer=tf.keras.regularizers.l2(0))(input_tensor)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)

            x = tf.keras.layers.Conv2D(filters[1], kernel_size, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0))(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)

            x = tf.keras.layers.Conv2D(filters[2], (1, 1), kernel_regularizer=tf.keras.regularizers.l2(0))(x)
            x = tf.keras.layers.BatchNormalization()(x)

            x = tf.keras.layers.Add()([x, input_tensor])
            x = tf.keras.layers.Activation('relu')(x)
            return x
        return helper
        
    def up_block(kernel_size, filters):
        def helper(input_tensor):

            x = tf.keras.layers.Conv2D(filters[0], (1, 1), kernel_regularizer=tf.keras.regularizers.l2(0))(input_tensor)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)

            x = tf.keras.layers.Conv2DTranspose(filters[1], 4, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0))(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)

            x = tf.keras.layers.Conv2D(filters[2], (1, 1), kernel_regularizer=tf.keras.regularizers.l2(0))(x)
            x = tf.keras.layers.BatchNormalization()(x)

            x_init = tf.keras.layers.Conv2DTranspose(filters[2], (1, 1), strides=2, kernel_regularizer=tf.keras.regularizers.l2(0), bias_initializer=None)(x)
            x = tf.keras.layers.Add()([x, x_init])
            x = tf.keras.layers.Activation('relu')(x)
            return x
        return helper
        
    original_shape = [input_shape[0], input_shape[1]]
    inp = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0))(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = conv_shortcut_block((3, 3), [64, 64, 256], strides=(1, 1))(x)
    x = block((3, 3), [64, 64, 256])(x)
    x = block((3, 3), [64, 64, 256])(x)
    x = conv_shortcut_block((3, 3), [128, 128, 512])(x)
    x = block((3, 3), [128, 128, 512])(x)
    x = block((3, 3), [128, 128, 512])(x)
    x = block((3, 3), [128, 128, 512])(x)
    x3 = x
    x = conv_shortcut_block(3, [256, 256, 1024])(x)
    x = block(3, [256, 256, 1024])(x)
    x = block(3, [256, 256, 1024])(x)
    x = block(3, [256, 256, 1024])(x)
    x = block(3, [256, 256, 1024])(x)
    x = block(3, [256, 256, 1024])(x)
    x4 = x
    x = conv_shortcut_block(3, [512, 512, 2048])(x)
    x = block(3, [512, 512, 2048])(x)
    x = block(3, [512, 512, 2048])(x)

    x5 = x
    x = attention(x)
    
    x5_2 = tf.keras.layers.Conv2DTranspose(num_class, 4, strides=2, padding='same', activation=None)(x5)
    x5_x4 = x5_2 + tf.keras.layers.Conv2D(num_class, 1, padding='same', activation='relu')(x4)
    
    x5_x4_2 = tf.keras.layers.Conv2DTranspose(num_class, 4, strides=2, padding='same', activation=None)(x5_x4)
    x5_x4_x3 = x5_x4_2 + tf.keras.layers.Conv2D(num_class, 1, padding='same', activation='relu')(x3)
    
    score = tf.keras.layers.Conv2DTranspose(num_class, 16, strides=8, padding='same', activation=None)(x5_x4_x3)
 
    score = tf.nn.softmax(score, axis=-1)
    return tf.keras.models.Model(inputs=inp, outputs=score)
        
def VGG16_FCN(input_shape, num_class):

    inp = tf.keras.layers.Input(shape=input_shape)
    
    original_shape = [input_shape[0], input_shape[1]]
    
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2())(inp)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2())(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2())(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2())(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2())(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2())(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2())(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2())(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2())(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2())(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2())(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2())(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2())(x)
    
    x = tf.keras.layers.Conv2D(4096, (7, 7), activation='relu', padding='same', dilation_rate=(2, 2), kernel_regularizer=tf.keras.regularizers.l2())(x) 
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Conv2D(4096, (1, 1), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2())(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.Conv2D(num_class, (1, 1), strides=(1, 1), kernel_initializer='he_normal', activation=None, padding='valid', kernel_regularizer=tf.keras.regularizers.l2())(x)
    
    x = tf.image.resize(x, [40, 40])
    x.set_shape((None, 40, 40, None))
    
    return tf.keras.models.Model(inputs=inp, outputs=x)