
import tensorflow as tf

def dice_coefficient(y_true, y_pred):
    smooth = 1.0  # スムージング項は数値の安定性を保つために使用されます
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)