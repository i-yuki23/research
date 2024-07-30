from tensorflow.keras.layers import Add, Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, MaxPooling3D, Conv3D, Dropout, GlobalAveragePooling3D
from tensorflow.keras.layers import BatchNormalization, Concatenate
from tensorflow.keras.optimizers import  Adam
import tensorflow as tf


# Define the kernel size as a constant
KERNEL_SIZE = (3, 3, 3)

def resnet_block(input_tensor, n_base, dilation_rate=1, strides=1, BN=True):
    x1 = Conv3D(n_base, kernel_size=KERNEL_SIZE, strides=strides, dilation_rate=dilation_rate, padding='same')(input_tensor)
    x2 = Conv3D(n_base, kernel_size=KERNEL_SIZE, strides=strides, dilation_rate=dilation_rate*2, padding='same')(input_tensor)

    x = Concatenate(axis=-1)([x1, x2])

    x = Conv3D(n_base, kernel_size=1, strides=1, padding='same')(x)
    if BN:
        x = BatchNormalization()(x)

    shortcut = input_tensor
    x = Add()([x, shortcut])
    x = Activation('elu')(x)
    return x


def ResNet_revised(n_base, input_shape, learning_rate, loss, metrics, class_num=1, dropout=0.4, BN=True, Sdropout="", resnet_block_num=6):
    inputs = Input(shape=input_shape)
    
    # Initial Conv3D layer
    x = Conv3D(n_base, kernel_size=1, strides=1, padding='same')(inputs)

    # ResNet blocks with dilated convolutions
    dilation_rates = [1, 2, 3]  # Example of using multiple dilation rates
    for i in range(resnet_block_num):
        dilation_rate = dilation_rates[i % len(dilation_rates)]
        x = resnet_block(x, n_base, strides=1, dilation_rate=dilation_rate, BN=BN)
        
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = GlobalAveragePooling3D()(x)

    # Dense and Dropout layers
    x = Dense(n_base * 8, activation='elu')(x)
    if dropout:
        x = Dropout(dropout)(x)
    x = Dense(n_base * 4, activation='elu')(x)
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
