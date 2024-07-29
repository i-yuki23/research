from tensorflow.keras.layers import Add, Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dense, Activation, MaxPooling3D, Conv3D, Dropout, GlobalAveragePooling3D
from tensorflow.keras.layers import BatchNormalization, Concatenate
from tensorflow.keras.optimizers import AdamW , Adam
import tensorflow as tf


# Define the kernel size as a constant
KERNEL_SIZE = (3, 3, 3)

def resnet_block(input_tensor, n_base, strides=1, BN=True):
    x1 = Conv3D(n_base, kernel_size=KERNEL_SIZE, strides=strides, dilation_rate=1, padding='same')(input_tensor)
    x2 = Conv3D(n_base, kernel_size=KERNEL_SIZE, strides=strides, dilation_rate=2, padding='same')(input_tensor)

    x = Concatenate(axis=-1)([x1, x2])

    x = Conv3D(n_base, kernel_size=1, strides=1, padding='same')(x)
    if BN:
        x = BatchNormalization()(x)

    shortcut = input_tensor
    x = Add()([x, shortcut])
    x = Activation('elu')(x)
    return x


def ResNet_revised(n_base, input_shape, learning_rate, loss, metrics, class_num=1, dropout=0.4, BN=True, Sdropout="", resnet_block_num=3):
    inputs = Input(shape=input_shape)
    
    # Initial Conv3D layer
    x = Conv3D(n_base, kernel_size=1, strides=1, padding='same')(inputs)
    
    # ResNet blocks
    for _ in range(resnet_block_num):
        x = resnet_block(x, n_base, strides=1, BN=BN)

    x = Conv3D(1, kernel_size=1, strides=1, padding='same')(x)

    # Dense and Dropout layers
    x = Flatten()(x)

    x = Dense(n_base, activation='elu')(x)
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
