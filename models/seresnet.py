import tensorflow.keras
from layers.seblock import SeResBlock

def build_neural_network_deep_arch(data_size_in, n_classes: int, verbose: int = 1) -> keras.Model:
      
    inputs = layers.Input(shape=data_size_in)
    
    init = tf.keras.initializers.HeNormal()
    reg = tf.keras.regularizers.l2(0.001)
    
    #x = layers.RandomFlip(mode='horizontal')(inputs)
    x = layers.RandomRotation(factor=0.1)(inputs)
    x = layers.RandomTranslation(height_factor=0.2, width_factor=0.2)(x)
    
    x = layers.Conv2D(64, kernel_size=7, strides=2, kernel_initializer=init, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activations.relu)(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)
    
    x = SeResBlock(x, 64, reg=reg, init=init)
    x = SeResBlock(x, 64, reg=reg, init=init)

    x = SeResBlock(x, 128, reg=reg, init=init)
    x = SeResBlock(x, 128, reg=reg, init=init)

    x = SeResBlock(x, 256, reg=reg, init=init)
    x = SeResBlock(x, 256, reg=reg, init=init)

    x = SeResBlock(x, 512, reg=reg, init=init)
    x = SeResBlock(x, 512, reg=reg, init=init)
    
    x = layers.AveragePooling2D(2)(x)
    x = keras.layers.Flatten()(x)

    x = layers.Dense(512, kernel_initializer=init, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activations.relu)(x)
    x = layers.Dropout(0.3)(x)
    
    predictions = layers.Dense(n_classes, activation='softmax', kernel_regularizer=reg)(x)
    
    model = keras.Model(inputs=inputs, outputs=predictions)
    
    return model