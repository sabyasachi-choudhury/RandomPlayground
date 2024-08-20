"""Imports"""
import time

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
import numpy as np
from matplotlib import pyplot as plt
print("gpus:", len(tf.config.list_physical_devices('GPU')))

"""Data Loading and preprocessing"""
(train_x, train_y), (_, __) = mnist.load_data()
display_stuff = False
train_x = np.reshape(train_x, [train_x.shape[0], 28, 28, 1]).astype('float32')
train_x /= 255.0
if display_stuff:
    print("Data shapes", train_x.shape, train_y.shape)
BATCH_SIZE = 256
BUFFER_SIZE = 60000
train_ds = tf.data.Dataset.from_tensor_slices(train_x).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

"""Model functions"""
# generator
def make_generator_model():
    model = models.Sequential([
        layers.Dense(7*7*256, use_bias=False, input_shape=(100,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Reshape([7, 7, 256]),
        layers.Conv2DTranspose(128, (5, 5), use_bias=False, strides=(1, 1), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(64, (5, 5), use_bias=False, strides=(2, 2), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(1, (5, 5), use_bias=False, strides=(2, 2), padding='same', activation='tanh')
    ])
    return model


# discriminator
def discriminator():
    model = models.Sequential([
        layers.Conv2D(64, kernel_size=(5, 5), strides=(1, 1), padding='same', input_shape=(28, 28, 1)),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.MaxPool2D([2, 2], strides=[2, 2]),
        layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.MaxPool2D([2, 2], strides=[2, 2]),
        layers.Flatten(),
        layers.Dense(1)
    ])
    return model


if display_stuff:
    seed = tf.random.normal([1, 100])
    gen = make_generator_model()
    im = gen(seed, training=False)
    plt.imshow(im[0, :, :, 0], cmap='gray')
    plt.show()

"""Loss functions"""
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# discriminator loss
def disc_loss(real_out, fake_out):
    real_loss = loss(tf.ones_like(real_out), real_out)
    fake_loss = loss(tf.zeros_like(fake_out), fake_out)
    total_loss = real_loss + fake_loss
    return total_loss


# generator loss
def gen_loss(fake_out):
    return loss(tf.ones_like(fake_out), fake_out)


"""optimizers and models"""
gen_opt, disc_opt = Adam(), Adam()
gen, disc = make_generator_model(), discriminator()


"""training loop requirements"""
@tf.function
def step(real_images):
    with tf.device('/GPU:0'):
        random_seed = tf.random.normal([BATCH_SIZE, 100])

        with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
            gen_images = gen(random_seed, training=True)
            real_output, fake_output = disc(real_images), disc(gen_images)
            generator_loss, discriminator_loss = gen_loss(fake_output), disc_loss(real_output, fake_output)

        gen_grad = gen_tape.gradient(generator_loss, gen.trainable_variables)
        disc_grad = disc_tape.gradient(discriminator_loss, disc.trainable_variables)

        gen_opt.apply_gradients(zip(gen_grad, gen.trainable_variables))
        disc_opt.apply_gradients(zip(disc_grad, disc.trainable_variables))
    return generator_loss

def train(dataset, epochs):
    mls = []
    for epoch in range(1, epochs+1):
        start = time.time()
        for batch in dataset:
            mls.append(step(batch))
        print(f"Epoch: {epoch}, gen_loss:{np.mean(mls)}, {time.time()-start} seconds")
    gen.save('generator')
    disc.save('discriminator')


print("start training")
train(train_ds, 50)

# model = models.load_model('generator')
# for x in range(10):
#     seed = tf.random.normal([BATCH_SIZE, 100])
#     gen_img = model(seed, training=False)
#     plt.imshow(gen_img[0, :, :, 0], cmap='gray')
#     plt.show()