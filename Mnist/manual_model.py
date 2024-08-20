# Imports
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# loading mnist dataset
"""Legend: train_x = pixels, or the individual imgs
            train_y = labels for each image
            test_x = testing img pixels
            test_y testing img labels"""
(train_x, train_y), (test_x, test_y) = mnist.load_data()


# getting training batches
def get_batch(x_train, y_train, size):
    indices = np.random.randint(0, len(y_train), size)
    return x_train[indices, :, :], y_train[indices]


# chit-chat vars
epochs = 10
batch_size = 100
h1_nodes = 300
h2_nodes = 10

# normalize all images. Normally more irritating, but simple here - just divide by 255 (max possible pixel value)
train_x = np.array(train_x, dtype=float)
test_x = np.array(test_x, dtype=float)
train_x /= 255.0
test_x /= 255.0

# make test_x a tf.Variable (Dunno why tho)
test_x = tf.Variable(test_x)

# Set weights and biases
# hl1
w1 = tf.Variable(tf.random.normal([784, h1_nodes], stddev=0.03), name="w1")
b1 = tf.Variable(tf.random.normal([h1_nodes]), name="b1")
# hl2
w2 = tf.Variable(tf.random.normal([h1_nodes, h2_nodes], stddev=0.03), name="w2")
b2 = tf.Variable(tf.random.normal([h2_nodes]), name="b2")


# feed forward function
def nn_model(x_input, weight_1=w1, bias_1=b1, weight_2=w2, bias_2=b2):
    # flatten input image from 28x28 to 28^2 or 784
    x_input = tf.reshape(x_input, (x_input.shape[0], -1))
    # first layer stuff
    # multiplying weights and adding biases
    x = tf.add(tf.matmul(tf.cast(x_input, tf.float32), weight_1), bias_1)
    # THE REST OF THE STUFF INSIDE FUNCTION IS CONFUSING
    # applying relu activation to the first hidden layer
    x = tf.nn.relu(x)
    logits = tf.add(tf.matmul(x, weight_2), bias_2)
    return logits


# loss function?
def loss_fn(logits, labels):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    return cross_entropy


# optimizer.
# using adam here, which is that gradient descent thing 3Blue1Brown showed
optimizer = tf.keras.optimizers.Adam()

print(get_batch(train_x, train_y, 3))


"""Finally, training!"""
# getting optimal number of batches
batches = int(len(train_y)/batch_size)
for ep in range(epochs):
    avg_loss = 0
    """Every epoch, you run through all the batches. That's why it's so slow"""
    for b in range(batches):
        batch_x, batch_y = get_batch(train_x, train_y, batch_size)
        batch_x = tf.Variable(batch_x)
        batch_y = tf.Variable(batch_y)
        batch_y = tf.one_hot(batch_y, 10)
        with tf.GradientTape() as tape:
            logits = nn_model(batch_x, w1, b1, w2, b2)
            loss = loss_fn(logits, batch_y)
        gradients = tape.gradient(loss, [w1, b1, w2, b2])
        optimizer.apply_gradients(zip(gradients, [w1, b1, w2, b2]))
        avg_loss += loss/batches
    test_logits = nn_model(test_x, w1, b1, w2, b2)
    max_prob = tf.argmax(test_logits, axis=1)
    test_acc = np.sum(max_prob.numpy() == test_y) / len(test_y)
    print(f"epoch: {ep + 1}")
    print(f"loss: {avg_loss:.3f}")
    print(f"accuracy: {test_acc*100:.3f}%")
    print("----------end epoch-----------")
print("end training!")