from tensorflow.keras.layers import LayerNormalization, Dense, Layer, Flatten, Activation, MaxPooling3D, Conv3D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# Define the kernel size as a constant
KERNEL_SIZE = (3, 3, 3)

class ConvNeXtBlock(Layer):
    def __init__(self, filters, drop_path=0.0, layer_scale_init_value=1e-6, **kwargs):
        super(ConvNeXtBlock, self).__init__(**kwargs)
        self.conv = Conv3D(filters, kernel_size=KERNEL_SIZE, padding='same')
        self.layer_norm = LayerNormalization(epsilon=1e-6)
        self.pointwise_conv1 = Conv3D(filters * 4, kernel_size=(1, 1, 1))
        self.gelu = Activation('gelu')
        self.pointwise_conv2 = Conv3D(filters, kernel_size=(1, 1, 1))
        self.drop_path = Dropout(drop_path) if drop_path > 0.0 else lambda x: x
        self.layer_scale = self.add_weight(
            shape=(filters,),
            initializer='zeros',
            trainable=True
        ) if layer_scale_init_value > 0 else None

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.layer_norm(x)
        x = self.pointwise_conv1(x)
        x = self.gelu(x)
        x = self.pointwise_conv2(x)
        if self.layer_scale is not None:
            x = self.layer_scale * x
        x = inputs + self.drop_path(x, training=training)
        return x

def ConvNeXt3D(input_shape, n_blocks, n_filters, learning_rate, loss, metrics, class_num=1, dropout=0.4):
    inputs = Input(shape=input_shape)
    
    # Initial Conv3D layer
    x = Conv3D(n_filters, kernel_size=KERNEL_SIZE, strides=1, padding='same')(inputs)
    
    # ConvNeXt Blocks
    for _ in range(n_blocks):
        x = ConvNeXtBlock(n_filters)(x)
    
    # Pooling and Dense layers
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = Flatten()(x)
    x = Dense(n_filters * 4, activation='relu')(x)
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

