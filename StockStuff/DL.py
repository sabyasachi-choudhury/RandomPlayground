import os
import time
import keras
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
import json

print("GPUS", len(tf.config.list_physical_devices()))
"""
ENSEMBLE IDEA

Instead of putting the loaded models into the final model itself, make two new datasets containing their
predictions for all data points.
Use these predictions as new training data. 
During inference/usage, just make individual predictions first, the use the ensemble part
"""

companies = ("Apple", "Taiwan Semiconductor", "AlphabetC", "AlphabetA", "Amazon", "Facebook", "Tesla", "Nvidia",
             "Visa", "JohnsonNJohnson", "Alibaba", "Walmart", "America Bank", "United Health", "Home Depot",
             "Mastercard", "Disney", "ProctorGamble", "Paypal", "Netflix", "Adobe", "Exxon", "Oracle")


def make_snapshot(size, dx, dy):
    output_data = [dx[i * size:(i + 1) * size] for i in range(int(len(dx) / size))]
    output_labels = [dy[(i + 1) * size] for i in range(int(len(dy) / size))]
    return np.array(output_data), np.array(output_labels)


def dataset_gen(size):
    data_x, data_y = [], []

    for company in companies:
        data = pd.read_csv(r"C:\Users\Sabyasachi\PycharmProjects\TestGroundTwo\StockStuff\Data"
                           + "\\" + company + ".csv")
        data = data["Mean"]
        for i, entry in enumerate(data):
            if np.isnan(entry):
                data[i] = (data[i - 1] + data[i + 1]) / 2
        if len(data) / size == int(len(data) / size):
            data[len(data)] = data[len(data) - 1]
            print(len(data))

        record, label = make_snapshot(size, data)
        data_x.extend(record)
        data_y.extend(label)

    data_x, data_y = np.array(data_x), np.array(data_y)
    indices = np.random.permutation(data_y.shape[0])
    demarcation = int(0.95 * len(data_y))
    train_indices, test_indices = indices[:demarcation], indices[demarcation:]

    tx, ty = data_x[train_indices], data_y[train_indices]
    t_x, t_y = data_x[test_indices], data_y[test_indices]
    return (tx, ty), (t_x, t_y)


def dataset_gen_2(size, xcols: (list, tuple), ycols: (list, tuple),
                  data_source: list = companies, path_base=r"StockStuff\Data",
                  special_snap=False):
    def clean(d):
        for i1, stamp in enumerate(d):
            for i2, point in enumerate(stamp):
                if np.isnan(point):
                    d[i1][i2] = np.mean((d[i1 - 1][i2], d[i1 + 1][i2]))
        if len(d) / size == int(len(d) / size):
            d = np.concatenate((d, np.array([d[-1]])))
        return d

    def dim_1_check(arr: np.ndarray):
        if arr.shape[-1] == 1:
            return arr.reshape([elem for elem in arr.shape[:-1]])
        else:
            return arr

    data_x, data_y = [], []
    for company in data_source:
        data = pd.read_csv(path_base + "\\" + company + ".csv", usecols=xcols).to_numpy()
        label_data = pd.read_csv(path_base + "\\" + company + ".csv", usecols=ycols).to_numpy()
        data, label_data = clean(data), clean(label_data)

        if not special_snap:
            records, labels = make_snapshot(size, data, label_data)
            data_x.extend(records)
            data_y.extend(labels)
        else:
            data_x.extend(data)
            data_y.extend(label_data)

    data_x, data_y = dim_1_check(np.array(data_x)), dim_1_check(np.array(data_y))

    indices = np.random.permutation(data_y.shape[0])
    demarcation = int(0.95 * len(data_y))
    train_indices, test_indices = indices[:demarcation], indices[demarcation:]

    tx, ty = data_x[train_indices], data_y[train_indices]
    t_x, t_y = data_x[test_indices], data_y[test_indices]
    return (tx, ty), (t_x, t_y)


"""dl part"""


def good_dense():
    nn = models.Sequential([
        layers.Dense(30, input_shape=[30, ], activation='relu'),
        layers.Dense(20, activation='relu'),
        layers.Dense(1)
    ])
    return nn


def another_good_dense():
    # OUTCOLS = MEAN
    nn = models.Sequential([
        layers.Flatten(input_shape=[10, len(['Mean', 'Low', 'High'])]),
        layers.Dense(20, activation='relu'),
        layers.Dense(1)
    ])
    return nn


def dense_2():
    """EVEN BETTER!!!! DONT DELETE MIGHT HAVE A WINNER!!!!!"""
    nn = models.Sequential([
        layers.Flatten(input_shape=[SIZE, len(IN_COLS)]),
        layers.Dense(10, activation='relu'),
        layers.Dense(20, activation='relu'),
        layers.Dense(1)
    ])
    return nn


def multi_output():
    nn = models.Sequential([
        layers.Flatten(input_shape=[SIZE, len(IN_COLS)]),
        layers.Dense(10, activation='relu'),
        layers.Dense(20, activation='relu'),
        layers.Dense(len(OUT_COLS))
    ])
    return nn


def single_output():
    nn = models.Sequential([
        layers.Flatten(input_shape=[SIZE, 1]),
        layers.Dense(10, activation='relu'),
        layers.Dense(20, activation='relu'),
        layers.Dense(1)
    ])
    print(nn.summary())
    return nn


def conv():
    nn = models.Sequential([
        layers.Input(shape=[SIZE, ]),
        layers.Reshape(target_shape=[SIZE, len(IN_COLS)]),

        layers.Conv1D(32, [5], [1], padding='same'),
        layers.ReLU(),

        layers.AveragePooling1D([2], [2]),
        layers.Conv1D(64, [3], [2], padding='same'),
        layers.ReLU(),

        layers.AveragePooling1D([2], [2]),
        layers.Conv1D(1, [3], [2], padding='same'),

        layers.Flatten(),
        layers.Dense(len(OUT_COLS))
    ])
    print(nn.summary())
    return nn


def ensemble_1():
    model = models.Sequential([
        layers.Input(shape=[3, ]),
        layers.Dense(10),
        layers.Dense(1)
    ])
    return model


def ensemble_2():
    nn = models.Sequential([
        layers.Input(shape=[len(IN_COLS), ]),
        layers.Dense(10, activation='relu'),
        layers.Dense(20, activation='relu'),
        layers.Dense(1)
    ])
    return nn


def train(model, epochs, save):
    with tf.device('/GPU:0'):
        model.summary()
        model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])
        model.fit(train_x, train_y, epochs=epochs, validation_data=(test_x, test_y), batch_size=BATCH_SIZE)
        model.save(r"C:\Users\Sabyasachi\PycharmProjects\TestGroundTwo\StockStuff\Models" + "\\" + save)
        with open(r"StockStuff\Models" + "\\" + save + r"\size.txt", "w") as file:
            file.write(str(SIZE))
        with open(r"StockStuff\Models" + "\\" + save + r"\usecols.json", "w") as file:
            json.dump({"Input": IN_COLS, "Output": OUT_COLS}, file)


def checkpoint_train(model, epochs, save):
    if not os.path.isdir(r"StockStuff\Models" + "\\" + save):
        os.mkdir(r"StockStuff\Models" + "\\" + save)

    def shuffle(dx, dy):
        indices = np.random.permutation(range(len(dx)))
        return dx[indices], dy[indices]

    def create_batches(dx, dy):
        dx, dy = shuffle(dx, dy)
        num_batches = int(len(dx) / BATCH_SIZE)
        out_x = [dx[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] for i in range(num_batches)]
        out_y = [dy[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] for i in range(num_batches)]
        return np.array(out_x), np.array(out_y)

    def val_loss_fn(p, t):
        loss_list = np.array([abs(p[i] - t[i]) for i, elem in enumerate(t)])
        return np.mean(loss_list)

    loss_obj = MeanAbsoluteError()
    opt_obj = Adam()

    def loss_fn(x, y, training, nn):
        y_ = nn(x, training=training)
        return loss_obj(y_true=y, y_pred=y_)

    @tf.function
    def step(x, y):
        with tf.device('/GPU:0'):
            with tf.GradientTape() as tape:
                loss = loss_fn(x, y, training=True, nn=model)
            nn_grad = tape.gradient(loss, model.trainable_variables)
            opt_obj.apply_gradients(zip(nn_grad, model.trainable_variables))
        return loss

    val_best, train_best = 100, 100
    print(model.summary())
    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss = tf.keras.metrics.Mean()
        batched_x, batched_y = create_batches(train_x, train_y)

        for batch_i in range(len(batched_y)):
            loss_val = step(batched_x[batch_i], batched_y[batch_i])
            train_loss.update_state(loss_val)

        val_loss = val_loss_fn(model(test_x), test_y)
        print(f"Epoch: {epoch}, Time: {time.time() - start}, Training Loss: {train_loss.result()}, "
              f"Validation Loss: {val_loss}")

        if val_loss < val_best:
            val_best = val_loss
            print("saving model for val loss improvement")
            model.save(r"C:\Users\Sabyasachi\PycharmProjects\TestGroundTwo\StockStuff\Models" + "\\"
                       + save + "\\" + "val")
        if train_loss.result() < train_best:
            train_best = train_loss.result()
            print("saving model for train loss improvement")
            model.save(r"C:\Users\Sabyasachi\PycharmProjects\TestGroundTwo\StockStuff\Models" + "\\" +
                       save + "\\" + "train")

        print('\n')

    with open(r"StockStuff\Models" + "\\" + save + r"\size.txt", "w") as file:
        file.write(str(SIZE))
    with open(r"StockStuff\Models" + "\\" + save + r"\usecols.json", "w") as file:
        json.dump({"Input": IN_COLS, "Output": OUT_COLS}, file)


"""Use High for ensembles, cuz why not. Then try low. Mean is pointless."""
SIZE = 10
BATCH_SIZE = 512
IN_COLS = ['High']
OUT_COLS = ['High']
print("Starting data gen")
(train_x, train_y), (test_x, test_y) = dataset_gen_2(size=SIZE, xcols=IN_COLS, ycols=OUT_COLS)
print("train_x", train_x.shape, "test_x", test_x.shape)
print("train_y", train_y.shape, "test_y", test_y.shape)

checkpoint_train(dense_2(), 100, 'irl_test_high')

# mod = ensemble()
# print(mod.summary())
