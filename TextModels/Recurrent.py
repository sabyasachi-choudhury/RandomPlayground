import tensorflow as tf

print("gpus:", len(tf.config.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(True)

with tf.device('/GPU:0'):
    a = tf.constant([[1, 2], [3, 4]])
    b = tf.constant([[5, 6], [2, 9]])
    c = tf.matmul(a, b)
print(c)