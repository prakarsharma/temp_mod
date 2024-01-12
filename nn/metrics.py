
import tensorflow as tf

def CRM(y_true, y_pred):
    # return keras.ops.sum(keras.ops.abs(y_true - y_pred))/ keras.ops.sum(y_true) # keras > 3
    return tf.math.divide(tf.math.reduce_sum(tf.math.abs(y_true - y_pred)), tf.math.reduce_sum(y_true))
