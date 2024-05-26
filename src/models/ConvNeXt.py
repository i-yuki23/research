from tensorflow.keras.layers import Layer, LayerNormalization, Dense, Activation, Conv3D, GlobalAveragePooling3D
import tensorflow_addons as tfa
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
import tensorflow as tf

# Define the kernel size as a constant
KERNEL_SIZE = (2, 2, 2)


# class ConvNeXtBlock(Layer):
#     def __init__(self, filters, drop_path=0.0, layer_scale_init_value=1e-6, **kwargs):
#         super(ConvNeXtBlock, self).__init__(**kwargs)
#         self.conv = Conv3D(filters, kernel_size=KERNEL_SIZE, padding='same')
#         self.layer_norm = LayerNormalization(epsilon=1e-6)
#         self.pointwise_conv1 = Conv3D(filters * 4, kernel_size=(1, 1, 1))
#         self.gelu = Activation('gelu')
#         self.pointwise_conv2 = Conv3D(filters, kernel_size=(1, 1, 1))
#         self.drop_path = Dropout(drop_path) if drop_path > 0.0 else None
#         self.layer_scale = self.add_weight(
#             shape=(filters,),
#             initializer='zeros',
#             trainable=True
#         ) if layer_scale_init_value > 0 else None

#     def call(self, inputs, training=False):
#         x = self.conv(inputs)
#         x = self.layer_norm(x)
#         x = self.pointwise_conv1(x)
#         x = self.gelu(x)
#         x = self.pointwise_conv2(x)
#         if self.layer_scale is not None:
#             x = self.layer_scale * x
#         if self.drop_path is not None:
#             x = self.drop_path(x, training=training)
#         x = inputs + x
#         return x

# def ConvNeXt3D(input_shape, n_base, learning_rate, loss, metrics, class_num=1, n_blocks=4, dropout=0.4, BN=False, Sdropout=False):
#     inputs = Input(shape=input_shape)
    
#     # Initial Conv3D layer
#     x = Conv3D(n_base, kernel_size=KERNEL_SIZE, strides=1, padding='same')(inputs)
    
#     # ConvNeXt Blocks
#     for _ in range(n_blocks):
#         x = ConvNeXtBlock(n_base)(x)
    
#     # Pooling and Dense layers
#     x = MaxPooling3D(pool_size=(2, 2, 2))(x)
#     x = Flatten()(x)
#     x = Dense(n_base * 4, activation='relu')(x)
#     if dropout:
#         x = Dropout(dropout)(x)
#     outputs = Dense(class_num, activation='sigmoid')(x)
    
#     model = Model(inputs, outputs)
    
#     optimizer = AdamW(learning_rate=learning_rate)
#     model.compile(loss=loss,
#                   optimizer=optimizer,
#                   metrics=metrics)
#     model.summary()
    
#     return model


class ConvNeXt_Block(Layer):
    def __init__(self, in_channels, out_channels, factor):
        super().__init__()

        #ConvNeXtブロック1層目
        self.layer_1 = Conv3D(in_channels, kernel_size=(4, 4, 4), strides=(1, 1, 1), padding='same', groups=in_channels, use_bias=False)
        
        #ConvNeXtブロック2層目
        #正規化をBatchNormalizationからLayerNormalizationへ変更しています。
        self.layer_2 = LayerNormalization(epsilon = 1e-6)
        #チャンネル数が4倍大きくなるInverted Bottleneck構造に変更しています。
        self.layer_3 = Conv3D(4 * out_channels, kernel_size = 1, strides = 1, padding = 'valid', use_bias = False)

        #ConvNeXtブロック3層目
        #活性化関数をReLUからGELUに変更しています。
        self.layer_4 = Activation(tf.nn.gelu)
        self.layer_5 = Conv3D(out_channels, kernel_size = 1, strides = 1, padding = 'valid', use_bias = False)

        self.layer_6 = LayerNormalization(epsilon = 1e-6)
        self.layer_7 = Activation('linear')

        self.shortcut = self.short_cut(in_channels, out_channels)
        #StochasticDepthでランダムにショートカットのみとしています。
        self.stochastic = tfa.layers.StochasticDepth(factor)

    def short_cut(self, in_channels, out_channels):
        #ショートカットとの残差出力の際にチャンネル数が異なる場合は、ショートカットと合わせます。
        if in_channels != out_channels:
            self.ln_sc = LayerNormalization()
            self.conv_sc = Conv3D(out_channels, kernel_size = 1, strides = 1, padding = 'same', use_bias = False)
            return self.conv_sc
        else:
            return lambda x: x

    def call(self, x):
        shortcut = self.shortcut(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.layer_6(x)
        x = self.layer_7(x)
        x = self.stochastic([x, shortcut])
        return x
        
class ConvNeXt(tf.keras.Model):
    def __init__(self, input_shape, output_dim):
        super().__init__()

        self.ln_pre = LayerNormalization(epsilon=1e-6)
        self.stem = Conv3D(16, kernel_size=2, strides=2, use_bias=False, input_shape=input_shape)

        # Stage 1
        self.stage_1 = [ConvNeXt_Block(16, 16, 0.1) for _ in range(1)]

        self.ln_1 = LayerNormalization(epsilon=1e-6)
        self.ds_1 = Conv3D(32, kernel_size=2, strides=2, use_bias=False)

        # Stage 2
        self.stage_2 = [ConvNeXt_Block(32, 32, 0.1) for _ in range(1)]

        self.ln_2 = LayerNormalization(epsilon=1e-6)
        self.ds_2 = Conv3D(64, kernel_size=2, strides=2, use_bias=False)

        # Stage 3
        self.stage_3 = [ConvNeXt_Block(64, 64, 0.2) for _ in range(1)]

        self.ln_3 = LayerNormalization(epsilon=1e-6)
        self.ds_3 = Conv3D(128, kernel_size=2, strides=2, use_bias=False)

        self.pooling = GlobalAveragePooling3D()
        self.ln_4 = LayerNormalization(epsilon = 1e-6)
        self.activation = Dense(output_dim, activation = 'sigmoid')


    def call(self, x):
        x = self.stem(self.ln_pre(x))
        for layer in self.stage_1:
            x = layer(x)
        x = self.ds_1(self.ln_1(x))
        for layer in self.stage_2:
            x = layer(x)
        x = self.ds_2(self.ln_2(x))
        for layer in self.stage_3:
            x = layer(x)
        x = self.ds_3(self.ln_3(x))
        # for layer in self.stage_4:
        #     x = layer(x)
        x = self.activation(self.ln_4(self.pooling(x)))
        
        return x

def ConvNeXt3D(input_shape, learning_rate, loss, metrics, class_num=1, n_blocks=4, dropout=0.4, BN=False, Sdropout=False, n_base=None):
    model = ConvNeXt(input_shape, class_num)
    model.build(input_shape=(None, *input_shape))
    optimizer = AdamW(learning_rate=learning_rate)
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)
    model.summary()
    
    return model