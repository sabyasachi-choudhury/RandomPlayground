import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
print("gpus:", len(tf.config.list_physical_devices('GPU')))

"""loading mnist"""
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

"""basic preprocessing"""
train_images = tf.reshape(train_images, (train_images.shape[0], 28, 28, 1))
test_images = tf.reshape(test_images, (test_images.shape[0], 28, 28, 1))
train_images = np.array(train_images, dtype=float)
test_images = np.array(test_images, dtype=float)
train_images /= 255.0
test_images /= 255.0

"""model architecture"""
model = Sequential([
    # Conv2D(32, [5, 5], [1, 1], padding='SAME', activation='relu', input_shape=(28, 28, 1)),
    # MaxPool2D([2, 2], [2, 2], padding='SAME'),
    # Conv2D(64, [5, 5], [1, 1], padding='SAME', activation='relu'),
    # MaxPool2D([2, 2], [2, 2], padding='SAME'),

    Conv2D(32, [5, 5], [1, 1], activation='relu', input_shape=(28, 28, 1)),
    MaxPool2D([2, 2], [2, 2]),
    Conv2D(64, [5, 5], [1, 1], activation='relu'),
    MaxPool2D([2, 2], [2, 2]),

    Flatten(),
    Dense(64, activation='relu'),
    Dense(10)
])

with tf.device('/GPU:0'):
    model.summary()
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    print("compiled")
    history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
    model.save("self_convolutional")

    results = model.evaluate(test_images, test_labels, verbose=2)
print(results)