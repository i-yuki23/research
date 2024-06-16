from tensorflow.keras.layers import Add, Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dense, Activation, MaxPooling3D, Conv3D, Dropout, GlobalAveragePooling3D
from tensorflow.keras.layers import BatchNormalization,SpatialDropout3D
from tensorflow.keras.optimizers import AdamW , Adam
import tensorflow as tf


# Define the kernel size as a constant
KERNEL_SIZE = (3, 3, 3)

def resnet_block(input_tensor, n_base, strides=1, BN=False):
    x = Conv3D(n_base, kernel_size=KERNEL_SIZE, strides=strides, padding='same')(input_tensor)
    if BN:
        x = BatchNormalization()(x)
    x = Activation(tf.nn.gelu)(x)

    x = Conv3D(n_base, kernel_size=KERNEL_SIZE, strides=1, padding='same')(x)
    if BN:
        x = BatchNormalization()(x)

    shortcut = input_tensor
    if strides != 1 or input_tensor.shape[-1] != n_base:
        shortcut = Conv3D(n_base, kernel_size=(1, 1, 1), strides=strides, padding='same')(shortcut)
        if BN:
            shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation(tf.nn.gelu)(x)
    return x


def ResNet(n_base, input_shape, learning_rate, loss, metrics, class_num=1, dropout=0.4, BN=True, Sdropout=""):
    inputs = Input(shape=input_shape)
    
    # Initial Conv3D layer
    x = Conv3D(n_base, kernel_size=KERNEL_SIZE, strides=1, padding='same')(inputs)
    if BN:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # ResNet blocks
    x = resnet_block(x, n_base, strides=1, BN=BN)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    x = resnet_block(x, n_base*2, strides=1, BN=BN)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    
    x = resnet_block(x, n_base*4, strides=1, BN=BN)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    
    # Global Average Pooling
    x = GlobalAveragePooling3D()(x)
    
    # Dense and Dropout layers
    x = Dense(n_base*4, activation='relu')(x)
    if dropout:
        x = Dropout(dropout)(x)
    outputs = Dense(class_num, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)
    model.summary()
    
    return model


def ResNet1(n_base, input_shape, learning_rate, loss, metrics, class_num=1, dropout=0.4, BN=True, Sdropout=""):
    inputs = Input(shape=input_shape)
    
    # Initial Conv3D layer
    x = Conv3D(n_base, kernel_size=KERNEL_SIZE, strides=1, padding='same')(inputs)
    if BN:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    
    # ResNet blocks
    x = resnet_block(x, n_base, strides=1, BN=BN)

    x = resnet_block(x, n_base*2, strides=1, BN=BN)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    
    x = resnet_block(x, n_base*4, strides=1, BN=BN)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    
    # Global Average Pooling
    x = GlobalAveragePooling3D()(x)
    
    # Dense and Dropout layers
    x = Dense(n_base*4, activation='relu')(x)
    if dropout:
        x = Dropout(dropout)(x)
    outputs = Dense(class_num, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    
    optimizer = AdamW(learning_rate=learning_rate)
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)
    model.summary()
    
    return model
