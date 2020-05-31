import tensorflow as tf


def Resnet50_FCN(input_shape, num_class):
    
    def conv_shortcut_block(kernel_size, filters, strides=(2, 2)):
        def helper(input_tensor):

            x = tf.keras.layers.Conv2D(filters[0], (1, 1), strides=strides, kernel_regularizer=tf.keras.regularizers.l2())(input_tensor)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)

            x = tf.keras.layers.Conv2D(filters[1], kernel_size, padding='same', kernel_regularizer=tf.keras.regularizers.l2())(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)

            x = tf.keras.layers.Conv2D(filters[2], (1, 1), kernel_regularizer=tf.keras.regularizers.l2())(x)
            x = tf.keras.layers.BatchNormalization()(x)

            shortcut = tf.keras.layers.Conv2D(filters[2], (1, 1), strides=strides, kernel_regularizer=tf.keras.regularizers.l2())(input_tensor)
            shortcut = tf.keras.layers.BatchNormalization()(shortcut)

            x = tf.keras.layers.Add()([x, shortcut])
            x = tf.keras.layers.Activation('relu')(x)
            return x
        return helper
        
    def block(kernel_size, filters):
        def helper(input_tensor):

            x = tf.keras.layers.Conv2D(filters[0], (1, 1), kernel_regularizer=tf.keras.regularizers.l2())(input_tensor)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)

            x = tf.keras.layers.Conv2D(filters[1], kernel_size, padding='same', kernel_regularizer=tf.keras.regularizers.l2())(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)

            x = tf.keras.layers.Conv2D(filters[2], (1, 1), kernel_regularizer=tf.keras.regularizers.l2())(x)
            x = tf.keras.layers.BatchNormalization()(x)

            x = tf.keras.layers.Add()([x, input_tensor])
            x = tf.keras.layers.Activation('relu')(x)
            return x
        return helper
        
    original_shape = [input_shape[0], input_shape[1]]
    inp = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_shortcut_block((3, 3), [64, 64, 256], strides=(1, 1))(x)
    x = block((3, 3), [64, 64, 256])(x)
    x = block((3, 3), [64, 64, 256])(x)

    x = conv_shortcut_block((3, 3), [128, 128, 512])(x)
    x = block((3, 3), [128, 128, 512])(x)
    x = block((3, 3), [128, 128, 512])(x)
    x = block((3, 3), [128, 128, 512])(x)

    x = conv_shortcut_block(3, [256, 256, 1024])(x)
    x = block(3, [256, 256, 1024])(x)
    x = block(3, [256, 256, 1024])(x)
    x = block(3, [256, 256, 1024])(x)
    x = block(3, [256, 256, 1024])(x)
    x = block(3, [256, 256, 1024])(x)

    x = conv_shortcut_block(3, [512, 512, 2048])(x)
    x = block(3, [512, 512, 2048])(x)
    x = block(3, [512, 512, 2048])(x)

    x = tf.keras.layers.Conv2D(num_class, (1, 1), kernel_initializer='he_normal', activation='linear', padding='valid', strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2())(x)

    x = tf.image.resize(x, [40, 40])
    x = tf.squeeze(x, axis=-1)
    x.set_shape((None, 40, 40))
    
    return tf.keras.models.Model(inputs=inp, outputs=x)
        
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
    
    x = tf.keras.layers.Conv2D(num_class, (1, 1), strides=(1, 1), kernel_initializer='he_normal', activation='linear', padding='valid', kernel_regularizer=tf.keras.regularizers.l2())(x)
    
    x = tf.image.resize(x, [40, 40])
    x = tf.squeeze(x, axis=-1)
    x.set_shape((None, 40, 40))
    
    return tf.keras.models.Model(inputs=inp, outputs=x)