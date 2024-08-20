# Imports
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# loading mnist data
(train_img, train_lbl), (test_img, test_lbl) = mnist.load_data()
# preprocessing
train_img = tf.reshape(train_img, (train_img.shape[0], -1))
train_img = np.array(train_img, dtype=float)
test_img = np.array(test_img, dtype=float)
train_img /= 255.0
test_img /= 255.0

# neural network architecture
model = Sequential([
    Dense(784, activation='relu', input_shape=(784, )),
    Dense(10)
])
# model.compile(loss='sparse-categorical-crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# For training I need to define a loss function, an optimizer, a batch creator, and a custom loop
# batching
def get_batch(images, labels, size):
    indices = np.random.randint(0, len(labels), size)
    return images[indices, :], tf.one_hot(labels[indices], 10)


# loss
# loss_obj = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
# def loss_fn(nn, x, y, training):
#     y_ = nn(x, training=training)
#     return loss_obj(y_true=y, y_pred=y_)
def loss_fn(logits, labels):
    t_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    return t_loss


# gradient descent
def grad(nn, inputs, correct_labels):
    with tf.GradientTape() as tape:
        logits = nn(inputs, training=True)
        loss = loss_fn(logits, correct_labels)
    return loss, tape.gradient(loss, model.trainable_variables)


# optimizer
optimizer = tf.keras.optimizers.Adam()
# other required vars
epochs = 10
batch_size = 500


def train(nn, train_x, train_y, b_size, eps):
    total_batches = int(len(train_y) / b_size)
    for ep in range(eps):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.CategoricalCrossentropy()
        for b in range(total_batches):
            img_batch, label_batch = get_batch(train_x, train_y, b_size)
            loss, gradient = grad(nn, img_batch, label_batch)
            optimizer.apply_gradients(zip(gradient, nn.trainable_variables))
            epoch_loss_avg.update_state(loss)
            epoch_accuracy.update_state(label_batch, nn(img_batch))
        print("Epoch: {}\nLoss: {}\n Accuracy: {}\n-----------".format(ep,
                                                                       epoch_loss_avg.result(),
                                                                       epoch_accuracy.result()))
    nn.save("self_mnist")


train(model, train_img, train_lbl, 100, 50)
# tests
# test_input = np.array([train_img[0]])
# testing_labels = np.array(train_lbl[0])
# testing_labels = tf.one_hot([testing_labels], 10)
# print(testing_labels)
# # print(testing_labels)
# logits = model.predict(test_input)
# # print(logits)
# # print(test_input)
# # print(model.predict(test_input))
# # print(grad(model, test_input, testing_labels)