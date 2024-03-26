from tensorflow.keras.layers import Dense, Flatten, Dense, Activation, MaxPooling3D, Conv3D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization,SpatialDropout3D
from tensorflow.keras.optimizers import Adam

def Alexnet(n_base, input_shape, learning_rate, metrics, loss, class_num=1, dropout=0.4, BN=False, Sdropout=False):
    K_SIZE = 2
    model = Sequential()
    model.add(Conv3D(filters=n_base,input_shape=input_shape, kernel_size=(K_SIZE,K_SIZE,K_SIZE),
    strides=(1,1,1), padding='same'))
    if BN:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    if Sdropout:
        model.add(SpatialDropout3D(0.1))
    model.add(MaxPooling3D(pool_size=(2,2,2)))
    model.add(Conv3D(filters= n_base *2, kernel_size=(K_SIZE,K_SIZE,K_SIZE), strides=(1,1,1), padding='same'))
    if BN:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    if Sdropout:
        model.add(SpatialDropout3D(0.1))
    # model.add(MaxPooling3D(pool_size=(2,2,2)))
    model.add(Conv3D(filters= n_base *4, kernel_size=(K_SIZE,K_SIZE,K_SIZE), strides=(1,1,1), padding='same'))

    if BN:
        model.add(BatchNormalization())

    model.add(Activation('relu'))

    if Sdropout:
        model.add(SpatialDropout3D(0.1))

    model.add(Conv3D(filters= n_base *4, kernel_size=(K_SIZE,K_SIZE,K_SIZE), strides=(1,1,1), padding='same'))

    if BN:
        model.add(BatchNormalization())

    model.add(Activation('relu'))

    if Sdropout:
        model.add(SpatialDropout3D(0.1))

    model.add(Conv3D(filters= n_base *2, kernel_size=(K_SIZE,K_SIZE,K_SIZE), strides=(1,1,1), padding='same'))

    if BN:
        model.add(BatchNormalization())

    model.add(Activation('relu'))

    if Sdropout:
        model.add(SpatialDropout3D(0.1))

    model.add(MaxPooling3D(pool_size=(2,2,2)))
    model.add(Flatten())
    model.add(Dense(64))
    
    if dropout:
        model.add(Dropout(dropout))
        
    model.add(Activation('relu'))

    model.add(Dense(64))
    if dropout:
        model.add(Dropout(dropout))
        
    model.add(Activation('relu'))

    model.add(Dense(class_num))
    model.add(Activation('softmax'))  

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss=loss,
    optimizer = optimizer,
    metrics=metrics)
    model.summary()
    return model
