import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras import models, layers
from matplotlib import pyplot
import json

"""loading data"""
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

"""basic preprocessing"""
train_images = np.array(train_images, dtype=float) / 255.0
test_images = np.array(test_images, dtype=float) / 255.0
# print(train_images.shape)
# print(test_images.shape)
# train_images = np.reshape(train_images, [50000, 32, 32, 3, 1])
# test_images = np.reshape(test_images, [10000, 32, 32, 3, 1])
# print(train_images.shape)
# print(test_images.shape)
# train_labels = tf.one_hot(train_labels, depth=10)
# test_labels = tf.one_hot(test_labels, depth=10)

"""model architectures"""


def vgg_dropout():
    def sequential():
        model = models.Sequential([
            layers.Conv2D(32, [3, 3], input_shape=(32, 32, 3), activation='elu', padding='SAME'),
            layers.Conv2D(32, [3, 3], activation='elu', padding='same'),
            layers.Conv2D(32, [2, 2], strides=[2, 2], activation='elu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),

            layers.Conv2D(64, [3, 3], activation='elu', padding='SAME'),
            layers.Conv2D(64, [3, 3], activation='elu', padding='same'),
            layers.Conv2D(64, [2, 2], strides=[2, 2], activation='elu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Conv2D(128, [3, 3], activation='elu', padding='SAME'),
            layers.Conv2D(128, [3, 3], activation='elu', padding='same'),
            layers.Conv2D(128, [2, 2], strides=[2, 2], activation='elu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),

            layers.Flatten(),
            # layers.Dense(128, activation='relu'),
            layers.Dense(10)
        ])
        return model

    def block(filters, drop_rate, in_data):
        conv1 = layers.Conv2D(filters, [3, 3], padding='same', activation='elu', use_bias=True,
                              bias_initializer=glorot_uniform)(in_data)
        conv2 = layers.Conv2D(filters, [3, 3], padding='same', activation='elu', use_bias=True,
                              bias_initializer=glorot_uniform)(conv1)
        add = layers.Add()([conv2, in_data])
        bn = layers.BatchNormalization()(add)
        drop = layers.Dropout(drop_rate)(bn)
        return drop

    def new_block(filters, drop_rate, in_data):
        vgg = layers.Conv2D(filters, [3, 3], padding='same', activation='elu', use_bias=True,
                            bias_initializer=glorot_uniform)(in_data)
        vgg = layers.Conv2D(filters, [3, 3], padding='same', activation='elu', use_bias=True,
                            bias_initializer=glorot_uniform)(vgg)
        add_block = block(filters, drop_rate, in_data)

        add = layers.Add()([add_block, vgg])
        bn = layers.BatchNormalization()(add)
        return bn

    inputs = tf.keras.Input(shape=[32, 32, 3])

    block0 = layers.Conv2D(32, [5, 5], padding='same', activation='elu', use_bias=True,
                           bias_initializer=glorot_uniform)(inputs)
    block0 = layers.Conv2D(32, [5, 5], padding='same', activation='elu', use_bias=True,
                           bias_initializer=glorot_uniform)(block0)
    block0 = layers.BatchNormalization()(block0)
    block0 = layers.Dropout(0.2)(block0)

    change1 = layers.Conv2D(64, [2, 2], [2, 2])(block0)
    block1 = block(64, 0.3, change1)
    change2 = layers.Conv2D(128, [2, 2], [2, 2], activation='elu')(block1)
    block2 = new_block(128, 0.4, change2)
    change3 = layers.Conv2D(256, [2, 2], [2, 2], activation='elu')(block2)
    block3 = new_block(256, 0.5, change3)
    change4 = layers.Conv2D(512, [2, 2], [2, 2])(block3)
    block4 = block(512, 0.6, change4)
    #
    # change1 = layers.Conv2D(64, [2, 2], [2, 2], activation='elu')(block0)
    # block1 = block(64, 0.3, change1)
    #
    # change2 = layers.Conv2D(128, [2, 2], [2, 2], activation='elu')(block1)
    # block2 = block(128, 0.4, change2)
    #
    # change3 = layers.Conv2D(256, [2, 2], [2, 2], activation='elu')(block2)
    # block3 = block(256, 0.5, change3)

    flat = layers.GlobalAvgPool2D()(block4)
    outputs = layers.Dense(512, activation='elu')(flat)
    outputs = layers.Dense(10)(outputs)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model


def experiment():
    def base_block(filters, shape, in_data, drop_rate):
        l1 = layers.Conv2D(filters, shape, padding='same', activation='elu')(in_data)
        drop1 = layers.Dropout(drop_rate)(l1)
        bn1 = layers.BatchNormalization()(drop1)
        add1 = layers.Add()([bn1, in_data])
        bn_add1 = layers.BatchNormalization()(add1)

        l2 = layers.Conv2D(filters, shape, padding='same', activation='elu')(bn_add1)
        drop2 = layers.Dropout(drop_rate)(l2)
        bn2 = layers.BatchNormalization()(drop2)
        add2 = layers.Concatenate()([bn2, bn1, in_data])
        bn_add2 = layers.BatchNormalization()(add2)

        l3 = layers.Conv2D(filters, shape, padding='same', activation='elu')(bn_add2)
        drop3 = layers.Dropout(drop_rate)(l3)
        bn3 = layers.BatchNormalization()(drop3)
        add3 = layers.Concatenate()([bn3, bn2, bn1, in_data])
        bn_add3 = layers.BatchNormalization()(add3)

        # pool = layers.MaxPool2D([2, 2])(bn_add3)

        return bn_add3

    def layer_90(filters, shape):
        seq = models.Sequential([
            layers.Conv2D(filters, shape, padding='same', activation='elu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters, shape, padding='same', activation='elu'),
            layers.BatchNormalization()
        ])
        return seq

    def transition(drop_rate, pool_size):
        seq = models.Sequential([
            layers.MaxPool2D(pool_size),
            layers.Dropout(drop_rate)
        ])
        return seq

    def change(target_dims, conv_size):
        seq = models.Sequential([
            layers.MaxPool2D([2, 2]),
            layers.Conv2D(target_dims, conv_size, activation='elu', padding='same'),
            layers.BatchNormalization()
        ])
        return seq

    def res_layer(filters, in_data):
        conv1 = layers.Conv2D(filters, [3, 3], padding='same', activation='relu')(in_data)
        bn1 = layers.BatchNormalization()(conv1)
        pool1 = layers.MaxPool2D([2, 2])(bn1)
        res_conv1 = layers.Conv2D(filters, [3, 3], padding='same', activation='relu')(pool1)
        res_bn1 = layers.BatchNormalization()(res_conv1)
        res_conv2 = layers.Conv2D(filters, [3, 3], padding='same', activation='relu')(res_bn1)
        res_bn2 = layers.BatchNormalization()(res_conv2)
        add = layers.Add()([res_bn2, pool1])
        return add

    inputs = tf.keras.Input(shape=[32, 32, 3])

    conv1 = layers.Conv2D(32, [5, 5], padding='same', activation='elu')(inputs)
    l1 = base_block(32, [3, 3], conv1, 0.2)
    change1 = change(64, [3, 3])(l1)
    l2 = base_block(64, [3, 3], change1, 0.3)
    change2 = change(128, [3, 3])(l2)
    l3 = base_block(128, [3, 3], change2, 0.4)
    change3 = layers.MaxPool2D([2, 2])(l3)
    flat = layers.GlobalAvgPool2D()(change3)
    flat = layers.Dense(500, activation='elu')(flat)
    outputs = layers.Dense(10)(flat)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model


# def parallel():
#
def dense_experiment():
    def block(filters, shape, in_data, drop_rate):
        b1 = layers.Conv2D(filters, shape, padding='same', activation='elu')(in_data)
        b1 = layers.Conv2D(filters, shape, padding='same', activation='elu')(b1)
        drop1 = layers.Dropout(drop_rate)(b1)
        add1 = layers.Add()([drop1, in_data])
        bn1 = layers.BatchNormalization()(add1)

        return bn1

    inputs = layers.Input(shape=[32, 32, 3])

    change0 = layers.Conv2D(64, [3, 3], padding='same', activation='elu')(inputs)
    change0 = layers.Conv2D(64, [3, 3], padding='same', activation='elu')(change0)
    change0 = layers.Dropout(0.2)(change0)
    change0 = layers.BatchNormalization()(change0)

    block1 = block(64, [3, 3], change0, 0.3)
    change1 = layers.Conv2D(128, [2, 2], [2, 2])(block1)

    block2 = block(128, [3, 3], change1, 0.4)
    change2 = layers.Conv2D(256, [2, 2], [2, 2])(block2)

    block3 = block(256, [3, 3], change2, 0.5)
    change3 = layers.Conv2D(512, [2, 2], [2, 2])(block3)

    block4 = block(512, [2, 2], change3, 0.5)

    flat = layers.GlobalAveragePooling2D()(block4)
    outputs = layers.Dense(10)(flat)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model


def graph(hist):
    time = range(len(hist['accuracy']))
    fig, ax = pyplot.subplots()
    ax.plot(time, hist['accuracy'], label='accuracy', color='green')
    ax.plot(time, hist['loss'], label='loss', color='red')
    ax.plot(time, hist['val_loss'], label='val_loss', color='orange')
    ax.plot(time, hist['val_accuracy'], label='val_accuracy', color='blue')
    ax.legend()
    pyplot.show()


def train(model, epochs, opt, loss, name, save=False, save_file='cifar10_VGG', data_gen=False):
    model.summary()
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

    if data_gen:
        generator = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1
        )
        it_train = generator.flow(train_images, train_labels)
        model.fit_generator(it_train,
                            validation_data=(test_images, test_labels),
                            epochs=epochs)
    else:
        history = model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=epochs)

    with open("conv_data.json") as file:
        train_data = json.load(file)
        train_data[name] = history.history
    with open("conv_data.json", "w") as file:
        json.dump(train_data, file, indent=2)

    if save:
        model.save(save_file)

    graph(history.history)
    for par in history.history.keys():
        print(par, ":", history.history[par])


train(model=dense_experiment(),
      epochs=20,
      opt=tf.keras.optimizers.Adam(),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      save_file="88.5_res",
      save=True,
      name="88.5_res",
      data_gen=True
      )
# mod = experiment()
# mod.summary()
# statistics variance, stddev, mean
"""Try base model for a lot more epochs"""
"""More Parameters - More training"""
"""Try bias"""

# with open("conv_data.json") as file:
#     data = json.load(file)
#
# print(models.load_model("experiment").evaluate(test_images, test_labels))
# graph(data["experiment"])
