from tensorflow.keras.layers import Dense, Flatten, Dense, Activation, MaxPooling3D, Conv3D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization,SpatialDropout3D
from tensorflow.keras.optimizers import Adam

def LeNet(n_base, input_shape, learning_rate, loss, metrics, class_num=1, dropout=0.4, BN=False, Sdropout=False):
    model = Sequential()
    model.add(Conv3D(n_base, kernel_size = (3, 3, 3), activation='relu', strides=1, padding='same', input_shape = input_shape))
    if BN:
        model.add(BatchNormalization())

    if Sdropout:
        model.add(SpatialDropout3D(0.1))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    
    
    model.add(Conv3D(n_base*2, kernel_size = (3, 3, 3), activation='relu', strides=1, padding='same'))
    if BN:
        model.add(BatchNormalization())
    if Sdropout:
        model.add(SpatialDropout3D(0.1))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    
    
    model.add(Flatten())
    model.add(Dense(n_base*2, activation='relu'))
    if dropout:
        model.add(Dropout(dropout))
    model.add(Dense(class_num, activation = 'sigmoid'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss=loss,
    optimizer = optimizer,
    metrics=metrics)    
    model.summary()
    
    return model