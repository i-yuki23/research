from tensorflow.keras.layers import Dense, Flatten, Dense, Activation, MaxPooling3D, Conv3D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

def VGG(input_shape, n_base, class_num, learning_rate , dropout=False, BN=False, Sdropout=False):

    model = Sequential()
    
    model.add(Conv3D(filters=n_base, input_shape=input_shape, kernel_size=(3,3,3), strides=(1,1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(filters=n_base, kernel_size=(3,3,3), strides=(1,1,1), padding='same')) 
    model.add(Activation('relu'))

    model.add(MaxPooling3D(pool_size=(2,2,2)))

    model.add(Conv3D(filters=n_base*2, kernel_size=(3,3,3), strides=(1,1,1), padding='same'))
    model.add(Activation('relu'))  
    model.add(Conv3D(filters= n_base*2, kernel_size=(3,3,3), strides=(1,1,1), padding='same')) 
    model.add(Activation('relu'))

    model.add(MaxPooling3D(pool_size=(2,2,2)))
    
    model.add(Conv3D(filters= n_base *4, kernel_size=(3,3,3), strides=(1,1,1), padding='same')) 
    model.add(Activation('relu'))
    model.add(Conv3D(filters= n_base *4, kernel_size=(3,3,3), strides=(1,1,1), padding='same')) 
    model.add(Activation('relu'))
    model.add(Conv3D(filters= n_base *4, kernel_size=(3,3,3), strides=(1,1,1), padding='same')) 
    model.add(Activation('relu'))
    
    model.add(MaxPooling3D(pool_size=(2,2,2)))

    model.add(Conv3D(filters= n_base *8, kernel_size=(3,3,3), strides=(1,1,1), padding='same')) 
    model.add(Activation('relu'))
    model.add(Conv3D(filters= n_base *8, kernel_size=(3,3,3), strides=(1,1,1), padding='same')) 
    model.add(Activation('relu'))
    model.add(Conv3D(filters= n_base *8, kernel_size=(3,3,3), strides=(1,1,1), padding='same')) 
    model.add(Activation('relu'))

    model.add(MaxPooling3D(pool_size=(2,2,2)))

    model.add(Conv3D(filters= n_base *8, kernel_size=(3,3,3), strides=(1,1,1), padding='same')) 
    model.add(Activation('relu'))
    model.add(Conv3D(filters= n_base *8, kernel_size=(3,3,3), strides=(1,1,1), padding='same')) 
    model.add(Activation('relu'))
    model.add(Conv3D(filters= n_base *8, kernel_size=(3,3,3), strides=(1,1,1), padding='same')) 
    model.add(Activation('relu'))

    model.add(MaxPooling3D(pool_size=(2,2,2)))
    
    model.add(Flatten())
    
    model.add(Dense(64))
    if dropout:
        model.add(Dropout(0.2))
    model.add(Activation('relu'))


    model.add(Dense(64))
    if dropout:
        model.add(Dropout(0.2))
    model.add(Activation('relu'))


    model.add(Dense(64))
    if dropout:
        model.add(Dropout(0.2))
    model.add(Activation('relu'))

    model.add(Dense(class_num))
    model.add(Activation('softmax'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy',
    optimizer = optimizer,
    metrics=["accuracy"])

    model.summary()
    
    return model
