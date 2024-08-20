import math
import random
import time
from matplotlib import pyplot as plt
import requests
import pandas as pd
import numpy as np
import json
import csv
import re
import os
import tensorflow as tf
from tensorflow.keras.losses import MeanAbsoluteError, SparseCategoricalCrossentropy, BinaryCrossentropy, \
    MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models, layers
import StockAnalysis2 as sk2
from StockAnalysis2 import sign_simulator, sign_simulator_candle

print("Imports done")
api_key = "ALTRNNOSCIYJAHZI"

# def store_data(symbol, file_name, call_type, size="full", data_type="csv", adjusted=False,
#                special=False, **kwargs):
#     # if call_type == "TIME_SERIES_INTRADAY":
#     #     url = "https://www.alphavantage.co/query?function={call_type}&symbol={symbol}&interval={period}&outputsize={size}&datatype={file_type}&apikey={api_key}"
#     # elif call_type == "TIME_SERIES_DAILY":
#     #     url = f"https://www.alphavantage.co/query?function={call_type}&symbol={symbol}&outputsize={size}&datatype={file_type}&apikey={api_key}"
#     # elif call_type == "TIME_SERIES_INTRADAY_EXTENDED":
#     #     url = f"https://www.alphavantage.co/query?function={call_type}&symbol={symbol}&interval={period}&outputsize={size}&apikey={api_key}"
#     # else:
#     #     raise Exception("EnterValidCallType")
#     num_array = []
#     for i in range(1, 3):
#         for j in range(1, 13):
#             num_array.append([i, j])
#     slices = ["year" + str(num[0]) + "month" + str(num[1]) for num in num_array]
#     query_parameters = {
#         "apikey": api_key,
#         "function": call_type,
#         "symbol": symbol,
#         "outputsize": size
#     }
#     if call_type == "TIME_SERIES_INTRADAY":
#         query_parameters["interval"] = kwargs["interval"]
#         query_parameters["adjusted"] = adjusted
#         query_parameters["datatype"] = data_type
#     elif call_type == "TIME_SERIES_INTRADAY_EXTENDED":
#         query_parameters["interval"] = kwargs["interval"]
#         query_parameters["slice"] =
#         query_parameters["adjusted"] = adjusted
#     elif call_type == "TIME_SERIES_DAILY":
#         query_parameters["datatype"] = data_type
#
#     url = "https://www.alphavantage.co/query?"
#     for key in query_parameters.keys():
#         url += key + "=" + str(query_parameters[key])
#         url += "&"
#     url = url[:-1]
#     r = requests.get(url, params=query_parameters)
#
#     def upper_first(string):
#         arr = list(string)
#         arr[0] = arr[0].upper()
#         return ''.join(arr)
#
#     if data_type == "csv":
#         csv_data = csv.reader(r.text.splitlines(), delimiter=',')
#         csv_data_df = []
#         for row in csv_data:
#             csv_data_df.append(row)
#         csv_data_df = np.array(csv_data_df)
#         csv_data_df = pd.DataFrame(
#             columns=[upper_first(header) for header in csv_data_df[0]],
#             data=csv_data_df[1:]
#         )
#         # print(csv_data_df)
#         csv_data_df.to_csv("StockStuff/model_playground/Data/" + file_name + ".csv")
#     elif data_type == "json":
#         data = r.json()
#         with open("StockStuff/model_playground/Data/" + file_name + ".json", "w") as file:
#             json.dump(data, file, indent=4)

"""Setup"""


def test_on_data(company):
    process_data = pd.read_csv("StockStuff/model_playground/Data/" + company + ".csv",
                               usecols=["timestamp", "high", "low"])
    saved_stamps = []
    regex = "(\S*)\s"
    for i, stamp in enumerate(process_data["timestamp"]):
        date_stamp = re.findall(regex, stamp)[0]
        if date_stamp not in saved_stamps:
            saved_stamps.append(date_stamp)
        else:
            process_data = process_data.drop([i])
    process_data = process_data.reset_index(drop=True)
    indices = list(process_data.index)
    indices.reverse()
    process_data = process_data.iloc[indices]
    process_data = process_data.reset_index(drop=True)
    return process_data


def spec(symbol, file_name, details=False):
    url = "https://www.alphavantage.co/query?"
    query_parameters = {
        "apikey": api_key,
        "function": "TIME_SERIES_INTRADAY_EXTENDED",
        "symbol": symbol,
        "outputsize": "full",
        "interval": "5min",
        "adjusted": "true",
    }
    num_array = []
    for i in range(1, 3):
        for j in range(1, 13):
            num_array.append([i, j])
    slices = ["year" + str(num[0]) + "month" + str(num[1]) for num in num_array]
    slices.pop(0)
    print(slices)

    for key in query_parameters.keys():
        url += key + "=" + str(query_parameters[key])
        url += "&"
    base_url = url

    def upper_first(string):
        arr = list(string)
        arr[0] = arr[0].upper()
        return ''.join(arr)

    def do_csv_stuff(link):
        r = requests.get(link)
        csv_data = csv.reader(r.text.splitlines(), delimiter=',')
        csv_data_df = []
        for row in csv_data:
            csv_data_df.append(row)
        csv_data_df = pd.DataFrame(
            columns=[upper_first(header) for header in csv_data_df[0]],
            data=csv_data_df[1:]
        )
        return csv_data_df

    base_df = do_csv_stuff(url + "slice=year1month1")
    if details:
        print(base_df)
    for data_slice in slices:
        time.sleep(15)
        print("\n", data_slice, "\n")
        url += "slice=" + data_slice
        try:
            new_df = do_csv_stuff(url)
            if details:
                print(new_df)
            url = base_url
            base_df = pd.concat([base_df, new_df])
        except ValueError or AssertionError:
            break

    base_df = base_df.reset_index(drop=True)
    indices = list(base_df.index)
    indices.reverse()
    base_df = base_df.iloc[indices]
    base_df = base_df.reset_index(drop=True)
    base_df.to_csv("StockStuff/model_playground/Data/" + file_name + ".csv")


companies = ["AAPL", "TSM", "GOOG", "GOOGL", "AMZN", "FB", "NVDA", "V", "JNJ", "BABA",
             "WMT", "BAC", "UNH", "HD", "MA", "DIS", "PG", "PYPL", "NFLX", "ADBE", "XOM", "ORCL",
             "TM", "KO", "CSCO", "TMO", "VZ"]


# spec("FCEL", "FCEL_VAL", details=True)
# spec("ATVI", "ATVI_VAL", details=True)
# spec("ATVI", "ATVI_VAL", details=True)

def plot(symbol: str, rows: list):
    data = pd.read_csv("StockStuff/model_playground/Data/" + symbol.upper() + "_NEW.csv", usecols=rows)
    for row in rows:
        plt.plot(data[row], label=row)
    plt.legend()
    plt.show()


def make_snapshot(size, dx, dy):
    output_data = [dx[i * size:(i + 1) * size] for i in range(int(len(dx) / size))]
    output_labels = [dy[(i + 1) * size] for i in range(int(len(dy) / size))]
    return np.array(output_data), np.array(output_labels)


def dataset_gen_2(size, xcols: (list, tuple), ycols: (list, tuple),
                  data_source: list = companies, path_base=r"StockStuff\model_playground\Data",
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
        data = pd.read_csv(path_base + "\\" + company + "_NEW" + ".csv", usecols=xcols).to_numpy()
        label_data = pd.read_csv(path_base + "\\" + company + "_NEW" + ".csv", usecols=ycols).to_numpy()
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


def directional_dataset_gen(size, x_cols: (list, tuple), y_cols: (list, tuple),
                            data_path: str = "StockStuff/model_playground/Data", ftse_15=False, shuffle=True,
                            low_data=True, mode='default', normalize=False):
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

    def get_directions(arr: np.ndarray):
        def get_num_dir(a, b):
            if b - a >= 0:
                return 1
            elif b - a < 0:
                return 0

        fin_arr = []
        for i in range(arr.shape[0] - 1):
            fin_arr.append(get_num_dir(arr[i], arr[i + 1]))
        fin_arr = np.array(fin_arr)
        return fin_arr

    def low_data_snap(d):
        output_data = [d[i * size:(i + 1) * size] for i in range(int(len(d) / size))]
        return output_data

    def make_snapshot(dx):
        output_data = [dx[i:size + i] for i in range(len(dx) - size)]
        return np.array(output_data)

    if not ftse_15:
        files = [file for file in os.listdir("StockStuff/model_playground/Data") if re.findall("NEW", file)]
    else:
        files = ["FTSE_15min_train.csv"]
    data_x, data_y = [], []
    for file in files:
        dx = pd.read_csv(data_path + "\\" + file, usecols=x_cols).to_numpy()
        dy = pd.read_csv(data_path + "\\" + file, usecols=y_cols).to_numpy()[size:]
        if normalize:
            dx /= np.max(dx)
        dx = clean(dx)
        if low_data:
            dx = low_data_snap(dx)
        else:
            dx = make_snapshot(dx)

        print(len(dx), len(dy))
        if mode == 'default':
            # directions = get_directions(np.array([elem[-1] for elem in data]))
            directions = get_directions(dy[size:])
        elif mode == 'float_y':
            directions = np.array([elem[-1] for elem in dy[1:]])
        elif mode == 'candle_y':
            directions = np.array([dy[i + 1][-1] - dy[i][-1] for i in range(len(dy) - 1)])
        dx = dx[:-1]
        data_x.extend(dx)
        data_y.extend(directions)
        print(file)
    data_x, data_y = dim_1_check(np.array(data_x)), dim_1_check(np.array(data_y))
    print(data_x.shape, data_y.shape)

    demarcation = int(0.95 * len(data_y))
    if shuffle:
        indices = np.random.permutation(data_y.shape[0])
    else:
        indices = list(range(data_y.shape[0]))
    train_indices, test_indices = indices[:demarcation], indices[demarcation:]

    tx, ty = data_x[train_indices], data_y[train_indices]
    t_x, t_y = data_x[test_indices], data_y[test_indices]
    return (tx, ty), (t_x, t_y)


def checkpoint_train(model, epochs, save, loss_obj, custom_val_loss=None):
    if not os.path.isdir(r"StockStuff\model_playground\Models" + "\\" + save):
        os.mkdir(r"StockStuff\model_playground\Models" + "\\" + save)

    with open(r"StockStuff\model_playground\Models" + "\\" + save + r"\size.txt", "w") as file:
        file.write(str(SIZE))
    with open(r"StockStuff\model_playground\Models" + "\\" + save + r"\usecols.json", "w") as file:
        json.dump({"Input": IN_COLS, "Output": OUT_COLS}, file)

    def shuffle(dx, dy):
        indices = np.random.permutation(range(len(dx)))
        return dx[indices], dy[indices]

    def create_batches(dx, dy):
        # dx, dy = shuffle(dx, dy)
        num_batches = int(len(dx) / BATCH_SIZE)
        out_x = [dx[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] for i in range(num_batches)]
        out_y = [dy[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] for i in range(num_batches)]
        return np.array(out_x), np.array(out_y)

    def val_loss_fn(p, t):
        loss_list = np.array([abs(p[i] - t[i]) for i, elem in enumerate(t)])
        return np.mean(loss_list)

    opt_obj = Adam()

    # model.compile(optimizer=opt_obj, metrics='acc', loss=loss_obj)

    def loss_fn(x, y, training, nn):
        y_ = nn(x, training=training)
        return loss_obj(y_true=y, y_pred=y_)

    @tf.function
    def step(x, y):
        with tf.device('/GPU:0'):
            with tf.GradientTape() as tape:
                loss = loss_fn(x, y, training=True, nn=model)
                print(loss)
            nn_grad = tape.gradient(loss, model.trainable_variables)
            opt_obj.apply_gradients(zip(nn_grad, model.trainable_variables))
        return loss

    val_best, train_best = 250, 250
    print(model.summary())
    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss = tf.keras.metrics.Mean()
        batched_x, batched_y = create_batches(train_x, train_y)

        for batch_i in range(len(batched_y)):
            loss_val = step(batched_x[batch_i], batched_y[batch_i])
            train_loss.update_state(loss_val)

        val_loss = val_loss_fn(model(test_x), test_y)
        val_loss_2 = None
        if custom_val_loss is not None:
            val_loss_2 = custom_val_loss(model)

        print(f"Epoch: {epoch}, Time: {time.time() - start}, Training Loss: {train_loss.result()}, "
              f"Validation Loss: {val_loss}, Secondary Val Loss: {val_loss_2}")

        if epoch == 1:
            model.save(r"C:\Users\Sabyasachi\PycharmProjects\TestGroundTwo\StockStuff\model_playground\Models" + "\\"
                       + save + "\\" + "val")
            model.save(r"C:\Users\Sabyasachi\PycharmProjects\TestGroundTwo\StockStuff\model_playground\Models" + "\\"
                       + save + "\\" + "train")
        if val_loss < val_best:
            val_best = val_loss
            print("saving model for val loss improvement")
            model.save(r"C:\Users\Sabyasachi\PycharmProjects\TestGroundTwo\StockStuff\model_playground\Models" + "\\"
                       + save + "\\" + "val")
        if train_loss.result() < train_best:
            train_best = train_loss.result()
            print("saving model for train loss improvement")
            model.save(r"C:\Users\Sabyasachi\PycharmProjects\TestGroundTwo\StockStuff\model_playground\Models" + "\\" +
                       save + "\\" + "train")

        print('\n')


def custom_checkpoint_train(model, epochs, save, loss_obj, loss_csv="LGVN_VAL", custom_val=True, normalize=False):
    if not os.path.isdir(r"StockStuff\model_playground\Models" + "\\" + save):
        os.mkdir(r"StockStuff\model_playground\Models" + "\\" + save)

    with open(r"StockStuff\model_playground\Models" + "\\" + save + r"\size.txt", "w") as file:
        file.write(str(SIZE))
    with open(r"StockStuff\model_playground\Models" + "\\" + save + r"\usecols.json", "w") as file:
        json.dump({"Input": IN_COLS, "Output": OUT_COLS}, file)

    def shuffle(dx, dy):
        indices = np.random.permutation(range(len(dx)))
        return dx[indices], dy[indices]

    def create_batches(dx, dy):
        # dx, dy = shuffle(dx, dy)
        num_batches = int(len(dx) / BATCH_SIZE)
        out_x = [dx[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] for i in range(num_batches)]
        out_y = [dy[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] for i in range(num_batches)]
        return np.array(out_x), np.array(out_y)

    def val_loss_fn(p, t):
        loss_list = np.array([abs(p[i] - t[i]) for i, elem in enumerate(t)])
        return np.mean(loss_list)

    opt_obj = Adam()

    model.compile(optimizer=opt_obj, metrics='acc', loss=loss_obj)

    def custom_val_loss(nn, data):
        global custom_ds
        capital = 3000
        held_quantity = 0
        if normalize:
            maximum = np.max(custom_ds)
            custom_ds /= maximum
        try:
            predictions = nn(custom_ds)
        except:
            predictions = nn(custom_ds.reshape([custom_ds.shape[0], custom_ds.shape[1]]))
        if normalize:
            custom_ds *= maximum
        for i in range(predictions.shape[0]):
            # Where model thinks the price will go
            try:
                current_price = data[i]
                price_at_trade = data[i]
                predicted_dir = predictions[i]
                if predicted_dir[0] > predicted_dir[1]:
                    predicted_dir = 0
                else:
                    predicted_dir = 1

                if predicted_dir == 0:
                    if held_quantity > 0:
                        capital += price_at_trade * held_quantity
                        held_quantity = 0
                else:
                    if held_quantity == 0 and price_at_trade != 0:
                        # Buy, if not already bought
                        try:
                            held_quantity = math.floor(capital / price_at_trade)
                            capital -= price_at_trade * held_quantity
                        except:
                            print(capital, price_at_trade)
            except:
                pass

        # At the end of trading
        capital += current_price * held_quantity
        return best_for_loss - capital

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

    loss_ds = pd.read_csv(f"StockStuff/model_playground/Data/{loss_csv}.csv", usecols=['Low']).to_numpy().flatten()

    val_best, train_best = 250, 250
    print(model.summary())
    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss = tf.keras.metrics.Mean()
        batched_x, batched_y = create_batches(train_x, train_y)

        for batch_i in range(len(batched_y)):
            loss_val = step(batched_x[batch_i], batched_y[batch_i])
            train_loss.update_state(loss_val)

        val_loss = val_loss_fn(model(test_x), test_y)
        val_loss_2 = custom_val_loss(model, loss_ds)

        print(f"Epoch: {epoch}, Time: {time.time() - start}, Training Loss: {train_loss.result()}, "
              f"Validation Loss: {val_loss}, Secondary Val Loss: {val_loss_2}")

        if epoch == 1:
            model.save(r"C:\Users\Sabyasachi\PycharmProjects\TestGroundTwo\StockStuff\model_playground\Models" + "\\"
                       + save + "\\" + "val")
            model.save(r"C:\Users\Sabyasachi\PycharmProjects\TestGroundTwo\StockStuff\model_playground\Models" + "\\"
                       + save + "\\" + "train")
        if val_loss < val_best:
            val_best = val_loss
            print("saving model for val loss improvement")
            model.save(r"C:\Users\Sabyasachi\PycharmProjects\TestGroundTwo\StockStuff\model_playground\Models" + "\\"
                       + save + "\\" + "val")
        if train_loss.result() < train_best:
            train_best = train_loss.result()
            print("saving model for train loss improvement")
            model.save(r"C:\Users\Sabyasachi\PycharmProjects\TestGroundTwo\StockStuff\model_playground\Models" + "\\" +
                       save + "\\" + "train")

        print('\n')


# plot("NFLX", ["High", "Close", "Low", "Open"])
# Try with longer periods for 5 min models
# Also check data around places with huge drops/lifts
"""Architectures"""

if True:
    def dense_1():
        model = models.Sequential([
            layers.Dense(20, input_shape=[20, ]),
            layers.Dense(20),
            layers.Dense(1)
        ])
        return model


    def direction_1():
        model = models.Sequential([
            layers.Dense(20, input_shape=[SIZE, ]),
            layers.Dense(20),
            layers.Dense(2)
        ])
        return model


    def direction_conv():
        # This one went fucking amazing
        model = models.Sequential([
            layers.Input(shape=[20, ]),
            layers.Reshape(target_shape=[20, 1]),
            layers.Conv1D(32, 5, 1),
            layers.Conv1D(64, 3, 1),
            layers.Flatten(),
            layers.Dense(2)
        ])
        model.summary()
        return model


    def direction_conv_sigma():
        model = models.Sequential([
            layers.Input(shape=[SIZE, ]),
            layers.Reshape(target_shape=[SIZE, 1]),
            layers.Conv1D(32, 5, 1),
            layers.Conv1D(64, 3, 1),
            layers.Flatten(),
            layers.Dense(1, activation='sigmoid')
        ])
        model.summary()
        return model


    def direction_conv_batchnorm():
        model = models.Sequential([
            layers.Input(shape=[SIZE, ]),
            layers.Reshape(target_shape=[SIZE, 1]),
            layers.Conv1D(32, 5, 1),
            layers.Dropout(0.1),
            layers.Conv1D(64, 3, 1),
            layers.Dropout(0.1),
            layers.Flatten(),
            layers.BatchNormalization(),
            layers.Dense(2)
        ])
        model.summary()
        return model


    def direction_conv_init():
        model = models.Sequential([
            layers.Input(shape=[SIZE, ]),
            layers.Reshape(target_shape=[SIZE, 1]),
            layers.Conv1D(32, 5, 1, kernel_initializer='he_uniform'),
            layers.Conv1D(64, 3, 1, kernel_initializer='he_uniform'),
            layers.Flatten(),
            layers.Dense(2, kernel_initializer='he_uniform')
        ])
        model.summary()
        return model


    def direction_conv_zero():
        # This one went fucking amazing
        model = models.Sequential([
            layers.Input(shape=[20, ]),
            layers.Reshape(target_shape=[20, 1]),
            layers.Conv1D(32, 5, 1, kernel_initializer='zeros'),
            layers.Conv1D(64, 3, 1, kernel_initializer='zeros'),
            layers.Flatten(),
            layers.Dense(2, kernel_initializer='zeros')
        ])
        model.summary()
        return model


    def direction_simple():
        model = models.Sequential([
            layers.Dense(20, input_shape=[SIZE, ], activation='elu'),
            layers.Dense(2)
        ])
        return model


    def direction_conv_simple():
        model = models.Sequential([
            layers.Reshape(target_shape=[SIZE, 1], input_shape=[SIZE, ]),
            layers.Conv1D(64, 3, 1, activation='relu'),
            layers.Flatten(),
            layers.Dense(2)
        ])
        return model


    def direction_lstm():
        model = models.Sequential([
            layers.Input(shape=[SIZE, ]),
            layers.Reshape(target_shape=[SIZE, 1]),
            # layers.Dense(32, activation='relu'),
            layers.LSTM(32),
            layers.Dense(2)
        ])
        return model


    def ignore_dense():
        to_ignore = models.load_model("StockStuff/model_playground/Models/directional_simple/val")
        to_ignore.summary()

        class Direction(tf.keras.Model):
            def __init__(self):
                super(Direction, self).__init__()

            def call(self, input_tensor):
                values = tf.math.maximum(input_tensor[:, 0], input_tensor[:, 1])
                values = tf.reshape(values, [tf.shape(input_tensor)[0], 1])
                directions = tf.where(tf.equal(input_tensor, values))[:, -1]
                return tf.cast(tf.reshape(directions, [tf.shape(values)[0], 1]), dtype=tf.float32)

        input_ = tf.keras.Input(shape=[SIZE, ])
        # reshape = layers.Reshape(target_shape=[SIZE, 1])(input_)
        # ignore_target = to_ignore(reshape)
        ignore_target = to_ignore(input_)

        ignore_target.trainable = False
        ignore_dir = Direction()(ignore_target)

        # concat = layers.Concatenate(axis=1)([reshape, ignore_dir])
        concat = layers.Concatenate(axis=1)([input_, ignore_dir])

        decider = layers.Dense(30, activation='relu')(concat)
        output_ = layers.Dense(2)(decider)
        model = tf.keras.Model(inputs=input_, outputs=output_)
        return model


    def ignore_conv():
        to_ignore = models.load_model("StockStuff/model_playground/Models/directional_conv/val")

        class Direction(tf.keras.Model):
            def __init__(self):
                super(Direction, self).__init__()

            def call(self, input_tensor):
                values = tf.math.maximum(input_tensor[:, 0], input_tensor[:, 1])
                values = tf.reshape(values, [tf.shape(input_tensor)[0], 1])
                directions = tf.where(tf.equal(input_tensor, values))[:, -1]
                return tf.cast(tf.reshape(directions, [tf.shape(values)[0], 1, 1]), dtype=tf.float32)

        input_ = tf.keras.Input(shape=[SIZE, ])
        reshape = layers.Reshape(target_shape=[SIZE, 1])(input_)
        ignore_target = to_ignore(reshape)
        ignore_target.trainable = False
        ignore_dir = Direction()(ignore_target)

        concat = layers.Concatenate(axis=1)([reshape, ignore_dir])
        decider_1 = layers.Conv1D(32, 5, 1, activation='relu')(concat)
        # decider_2 = layers.Conv1D(64, 3, 1, activation='relu')(decider_1)

        flatten = layers.Flatten()(decider_1)
        output_ = layers.Dense(2)(flatten)
        model = tf.keras.Model(inputs=input_, outputs=output_)
        return model


    def ensemble():
        model1 = models.load_model("StockStuff/model_playground/Models/ignore_dense/train")
        model2 = models.load_model("StockStuff/model_playground/Models/directional_simple/val")
        _input = tf.keras.Input(shape=[SIZE, ])
        # The two models
        lane1 = model1(_input)
        lane2 = model2(_input)
        lane1.trainable = False
        lane2.trainable = False
        # Concatenating model outputs, and reshaping them to (n, ) for future concat
        concat = layers.Concatenate()([lane1, lane2])
        reshape = layers.Reshape(target_shape=[SIZE, 1])(_input)
        # Completely separate branch. Extract features from data, and condense it to one channel
        feature_extractor = layers.Conv1D(32, 5, 1)(reshape)
        condenser = layers.Conv1D(1, 3, 1)(feature_extractor)
        # Reshape feature map to (n, ) for concat with model output
        reshape2 = layers.Reshape(target_shape=[14, ])(condenser)
        # concat with model output
        concat2 = layers.Concatenate()([reshape2, concat])
        # Final dense to decide
        decider = layers.Dense(22, activation='relu')(concat2)
        _output = layers.Dense(2)(decider)

        model = tf.keras.Model(inputs=_input, outputs=_output)
        return model


# customised for directional stuff
def find_best(data):
    capital = 3000
    held_quantity = 0
    predictions = np.array([np.random.choice(2, p=[0.53, 0.47]) if data[i + 1] - data[i] < 0
                            else np.random.choice(2, p=[0.47, 0.53]) for i in range(len(data) - 1)])
    for i in range(predictions.shape[0]):
        # Where model thinks the price will go
        current_price = data[i]
        price_at_trade = data[i]
        predicted_dir = predictions[i]
        if predicted_dir == 0:
            if held_quantity > 0:
                capital += price_at_trade * held_quantity
                held_quantity = 0
        else:
            if held_quantity == 0:
                # Buy, if not already bought
                held_quantity = math.floor(capital / price_at_trade)
                capital -= price_at_trade * held_quantity

    # At the end of trading
    capital += current_price * held_quantity
    return capital


# class LargeJumpLoss(tf.keras.losses.Loss):
#     def __init__(self):
#         super(LargeJumpLoss, self).__init__()
#
#     """For this one, use future costs as y_true
#     However, it is still a directional model
#     KK, probably will have to use candle y_true"""
#
#     def call(self, y_true, y_pred):
#         # true_profit = quantity * abs(batch[-1] - y_true)
#         # if y_pred[1] > y_pred[0]:
#         #     pred_profit = quantity * (y_true - batch[-1])
#         # else:
#         #     pred_profit = quantity * (batch[-1] - y_true)
#         # if not to_square:
#         #     return true_profit - pred_profit
#         # else:
#         #     return (true_profit - pred_profit) ** 2
#         quantity = tf.constant(50, dtype=tf.float32)
#         # max_profit = abs(y_true) * quantity
#         max_profit = tf.math.multiply(
#             tf.cast(tf.math.abs(y_true), dtype=tf.float32),
#             quantity
#         )
#         if (y_pred[0] > y_pred[1] and y_true > 0) or (y_pred[1] > y_pred[0] and y_true < 0):
#             print("aPPLe")
#             return tf.cast(
#                 tf.math.multiply(
#                     tf.constant(2, dtype=tf.float32),
#                     max_profit
#                 ),
#                 dtype=tf.float32
#             )
#         else:
#             print("bawdb")
#             return tf.constant([0]*BATCH_SIZE, dtype=tf.float32)
def loss_creator(from_logits=True, clip=True, profit_k=tf.constant(100.0, dtype=tf.float32)):
    # do direction and other stuff
    def custom_loss(y_true, y_pred):
        # Flipped this one over!!!!
        y_true_dirs = tf.one_hot(tf.where(y_true >= 0, 1, 0), depth=2, on_value=1, off_value=0, dtype=tf.float32)
        y_true = tf.cast(tf.abs(y_true), dtype=tf.float32)
        y_true = tf.reshape(y_true, shape=(y_true.shape[0], 1))
        if from_logits:
            y_pred = tf.nn.softmax(y_pred)
        if clip:
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        returns = -tf.math.reduce_sum(y_true_dirs * y_true * tf.math.log(y_pred) * profit_k, [1])
        # returns = tf.where(tf.math.is_nan(returns), tf.constant(0.0, dtype=tf.float32), returns)
        returns = tf.reduce_mean(returns)
        return returns

    return custom_loss


"""Formalities"""
SIZE = 20
BATCH_SIZE = 128
IN_COLS = ['Close']
OUT_COLS = ['Open']
# LOSS_OBJ = loss_creator(from_logits=True, clip=True, profit_k=1000)
# LOSS_OBJ = SparseCategoricalCrossentropy()
LOSS_OBJ = SparseCategoricalCrossentropy(from_logits=True)
# LOSS_OBJ = BinaryCrossentropy(from_logits=True)
print("Starting data gen")

# print(tf.nn.softmax(tf.convert_to_tensor([1, 100], dtype=tf.float32)))
# st = time.perf_counter()
# x = LOSS_OBJ(y_true=tf.constant([0.08, 0.35, -0.13, -0.41]),
#              y_pred=tf.constant([
#                  [0, 0],
#                  [3.18, 0.44],
#                  [1.1, 100.10],
#                  [0.85, 0.99]
#              ]))
# print(time.perf_counter() - st)
# print(x)
# print(SparseCategoricalCrossentropy(from_logits=False)(
#     y_true=tf.constant([0, 0, 1, 0]),
#     y_pred=tf.nn.softmax(tf.constant([
#         [0, 0],
#         [3.18, 0.44],
#         [1.1, 100.10],
#         [0.85, 0.99]
#     ]))
# ))

if True:
    custom_ds = pd.read_csv("StockStuff/model_playground/Data/XOM_NEW.csv", usecols=['Low']).to_numpy().flatten()
    custom_ds = sk2.dataset_gen_2(SIZE, IN_COLS, "StockStuff/model_playground/Data/XOM_NEW")
    # best_for_loss = find_best(custom_ds)
    best_for_loss = 50000
    print("XOM", best_for_loss)

    (train_x, train_y), (test_x, test_y) = directional_dataset_gen(SIZE, x_cols=IN_COLS, y_cols=OUT_COLS,
                                                                   shuffle=True, mode='default', normalize=False, low_data=False)
    print("train_x", train_x.shape, "test_x", test_x.shape)
    print("train_y", train_y.shape, "test_y", test_y.shape)
    print(train_x[0], train_x[1])
    print(train_y[0])
    #
    # # checkpoint_train(dense_1(), 200, "low_to_low_2")
    #
    model_name = "close_to_open"
    # custom_checkpoint_train(direction_simple(), 100, model_name, LOSS_OBJ)
    custom_checkpoint_train(direction_conv(), 10, model_name, LOSS_OBJ, normalize=False)

# d = pd.read_csv("StockStuff/ExtraData/FTSE_15min_train.csv")
# length = d.shape[0]
# train, test = d.iloc[:int(0.85*length)], d.iloc[int(0.85*length):]
# test.to_csv("StockStuff/ExtraData/FTSE_15min_val.csv")
# train.to_csv("StockStuff/ExtraData/FTSE_15min_train.csv")

# a = tf.constant([0, 1], dtype=tf.float16)
# b = tf.constant([1, 10], tf.float16)
# b = tf.nn.softmax(b)
# print(b)
# l = tf.reduce_mean(-tf.reduce_sum(a * tf.math.log(b)))
# print(l)

"""Don't use train for testing"""
"""Well, only thing left is reinforcement. """

if __name__ == "ain__":

    datasets = [file[:-4] for file in os.listdir("StockStuff/model_playground/Data") if re.findall("VAL", file)]
    stats_record = []
    raw_stats = [[], [], []]
    p1, p2, p3 = "StockStuff/model_playground/Models/close_to_open", \
                 "StockStuff/model_playground/Models/close_to_open", \
                 "StockStuff/model_playground/Models/ignore_dense"
    m1, m2, m3 = models.load_model(f"{p1}/val", compile=False), \
                 models.load_model(f"{p2}/train", compile=False), \
                 models.load_model(f"{p3}/train", compile=False)
    name1, name2, name3 = "val", "train", "best"
    for dataset in datasets:
        print(dataset)
        try:
            print(name1)
            cap_a = sign_simulator("StockStuff/model_playground/Data/" + dataset,
                                   model_path=p1,
                                   model=m1, capital=3000, interval=2500, emergency=False, to_graph=False, normalize=True)
            raw_stats[0].append(cap_a)
            print(
                "------------------------------------------------------------------------------------------------------")
            print('\n')
            print(name2)
            cap_b = sign_simulator("StockStuff/model_playground/Data/" + dataset,
                                   model_path=p2,
                                   model=m2, capital=3000, interval=2500, emergency=False, to_graph=False, normalize=True)
            raw_stats[1].append(cap_b)
            stats_record.append((cap_a - cap_b) / cap_b)
            print((cap_a - cap_b) / cap_b)
            print(
                "------------------------------------------------------------------------------------------------------")
            print('\n')
            print(name3)
            cap_c = sign_simulator("StockStuff/model_playground/Data/" + dataset,
                                   model_path=p3,
                                   model=m3, capital=3000, interval=2500, emergency=False, to_graph=False)
            raw_stats[2].append(cap_c)
        except:
            continue

    print(f"raw average, {name1} is better than {name2} by {sum(stats_record) / len(stats_record) * 100}, %")
    print(
        f"adjusted average, {name1} is better than {name2} by {sum(stats_record) - max(stats_record) - min(stats_record) / len(stats_record) * 100}, %")
    plt.plot(raw_stats[0], label=name1)
    plt.plot(raw_stats[1], label=name2)
    plt.plot(raw_stats[2], label=name3)
    plt.title("Plot of capitals")
    plt.legend()
    plt.show()
    #  ignorer dense train is best so far
    # But direction simple seems better than directional conv.
    # Gotta integrate it into another ignorer

"""TO DO!
Fuck around with LSTMs
Also, yknow, understand them
Use differences data, and train model to ignore model"""
"""Mack custom ignorer for candles"""
"""More importantly
IDEK, but, maybe make custom loss/val functions, which try to maximize capital??????"""

"""K, now dod it for train"""
"""Realization: Previous plan for loss won't work for training, but is pretty neat for validation"""
"""Idea for training: Try and make it put importance on predicting large jumps."""
