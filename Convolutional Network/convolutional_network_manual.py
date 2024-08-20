import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_labels = tf.one_hot(train_labels, depth=10)

# optimization vars
tf.compat.v1.disable_eager_execution()
learning_rate = 0.0001
epochs = 10
batch_size = 50

# placeholders
x = tf.compat.v1.placeholder(tf.float32, [None, 784])
x_shaped = tf.reshape(x, [-1, 28, 28, 1])
y = tf.compat.v1.placeholder(tf.float32, [None, 10])


# get batch function
def get_batch(size):
    indices = np.random.randint(0, len(train_labels), size)
    return train_images[indices, :, :], train_labels[indices]


# convolutional filters
# the function first
def create_conv_layer(input_data, num_channels, num_filters, filter_shape, pool_shape):
    """Filter shape
    Conv2d accepts 4D inputs (img_height, img_width, entering_channels, exiting_channels)"""
    conv_filter_shape = [filter_shape[0], filter_shape[1], num_channels, num_filters]

    # Weights and Biases
    weights = tf.Variable(tf.random.truncated_normal(conv_filter_shape, stddev=0.03))
    bias = tf.Variable(tf.random.truncated_normal([num_filters], stddev=0.03))

    # The actual convolution layer
    """Note, third arg is stride length.
    arg[1] and arg[2] are x and y strides. arg[0], arg[3] are moving between channels and batches, and are always 1
    to stay in one place
    Also, no need to give the input_shape - it's automatically determined via weight's shape"""
    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')

    # adding bias
    out_layer += bias

    # applying activation
    out_layer = tf.nn.relu(out_layer)

    """Now pooling
    Same like conv2d. arg[1] and [2] are x and y, rest should be 1 to stay in one place
    Same applies to strides"""
    k_size = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=k_size, strides=strides, padding="SAME")

    return out_layer


# Now creating layers as needed
layer1 = create_conv_layer(x_shaped, 1, 32, [5, 5], [2, 2])
layer2 = create_conv_layer(layer1, 32, 64, [5, 5], [2, 2])

# the traditional nn
flattened = tf.reshape(layer2, [-1, 7 * 7 * 64])

# Dense 1 stuff
wd1 = tf.Variable(tf.random.truncated_normal([7 * 7 * 64, 1000], stddev=0.03))
bd1 = tf.Variable(tf.random.truncated_normal([1000], stddev=0.01))
dense1 = tf.matmul(flattened, wd1) + bd1
dense1 = tf.nn.relu(dense1)

# Dense 2 stuff
wd2 = tf.Variable(tf.random.truncated_normal([1000, 10], stddev=0.03))
bd2 = tf.Variable(tf.random.truncated_normal([10], stddev=0.01))
dense2 = tf.matmul(dense1, wd2) + bd2
_y = tf.nn.softmax(dense2)

# cross_entropy, or loss
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense2, labels=y))

"""training"""
# optimizer
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# accuracy assessment
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialization operator
init_op = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init_op)
    total_batch = int(len(train_labels)/batch_size)
    for epoch in range(epochs):
        avg_cost = 0
        for b in range(total_batch):
            img_batch, label_batch = get_batch(batch_size)
            _, c = sess.run([optimizer, cross_entropy], feed_dict={x: img_batch, y: label_batch})
            avg_cost += c/total_batch
        test_acc = sess.run(accuracy, feed_dict={x: test_images, y: test_labels})
        print("Epoch:", (epoch+1), " cost:", "{:3f}".format(avg_cost), " accuracy:", "{:3f}".format(test_acc))

    print("\ntraining complete!")
    print(sess.run(accuracy, feed_dict={x: test_images, y: test_labels}))