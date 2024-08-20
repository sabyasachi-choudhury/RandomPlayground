import math
import random
import re
import time
import os
import pandas as pd
import numpy as np
# import re
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, losses
import matplotlib.pyplot as plt
import json


def day_test():
    data_1 = pd.read_csv(r"StockStuff\FTSE\Data 1.csv")
    t_date = "20/12/1999"

    date_df = pd.DataFrame(
        {"High": [data_1["High"][i] for i, date in enumerate(data_1["Date"]) if date == t_date],
         "Low": [data_1["Low"][i] for i, date in enumerate(data_1["Date"]) if date == t_date],
         "Open": [data_1["Open"][i] for i, date in enumerate(data_1["Date"]) if date == t_date],
         "Close": [data_1["Close"][i] for i, date in enumerate(data_1["Date"]) if date == t_date]}
    )

    data_2 = pd.read_csv(r"StockStuff\Data\Apple.csv")

    comp_df = pd.DataFrame(
        {"High": [data_2["High"][i] for i in range(data_2.shape[0] - date_df.shape[0], data_2.shape[0])],
         "Low": [data_2["Low"][i] for i in range(data_2.shape[0] - date_df.shape[0], data_2.shape[0])],
         "Open": [data_2["Open"][i] for i in range(data_2.shape[0] - date_df.shape[0], data_2.shape[0])],
         "Close": [data_2["Close"][i] for i in range(data_2.shape[0] - date_df.shape[0], data_2.shape[0])]}
    )


"""---------------------------------------------------------------------------------------------------------------"""

train_codes = ["AAPL", "TSM", "GOOG", "GOOGL", "AMZN", "FB", "TSLA", "NVDA", "V", "JNJ", "BABA",
               "WMT", "BAC", "UNH", "HD", "MA", "DIS", "PG", "PYPL", "NFLX", "ADBE", "XOM", "ORCL"]

train_companies = ["Apple", "Taiwan Semiconductor", "AlphabetC", "AlphabetA", "Amazon", "Facebook", "Tesla", "Nvidia",
                   "Visa", "JohnsonNJohnson", "Alibaba", "Walmart", "America Bank", "United Health", "Home Depot",
                   "Mastercard", "Disney", "ProctorGamble", "Paypal", "Netflix", "Adobe", "Exxon", "Oracle"]

val_codes = ["TM", "KO", "CSCO", "TMO", "NVO", "VZ"]
val_companies = ["Toyota", "CocaCola", "Cisco", "ThermoFisher", "Novo", "Verizon"]

all_codes = train_codes
all_codes.extend(val_codes)
all_companies = train_companies
all_companies.extend(val_companies)


def graph(df, title, show=True):
    metrics = ["High", "Low"]
    plt.figure()
    for m in metrics:
        plt.plot(df[m], label=m)
    plt.legend()
    plt.title(title)
    if show:
        plt.show()


def variation(in_df, method: str, k_size=30, metric="High"):
    def create_var_list(entry: [list, np.ndarray, pd.Series]):
        output = [abs(entry[i] - entry[i + 1]) for i in range(len(entry) - 1)]
        return output

    df = in_df[metric].to_numpy()
    if method == "kernel":
        limit = len(df) - k_size + 1
        var_means = [np.mean(create_var_list(df[i: i + k_size])) for i in range(limit)]
        return np.array(var_means)
    elif method == "section":
        limit = int(len(df) / k_size)
        var_means = [[np.mean(create_var_list(df[k_size * i: k_size * (i + 1)]))] * (k_size - 1) for i in range(limit)]
        return np.array(var_means).flatten()
    else:
        print("Gimme proper method")


def plots(path, metrics=None):
    if metrics is None:
        metrics = ["High", "Low", "Mean"]
    data = pd.read_csv(path, usecols=metrics)
    print(data.head())

    for m in metrics:
        plt.plot(data[m], label=m)
    plt.legend()
    plt.show()


"""------------------------------------------------------------------------------------------------------------------"""


# data = pd.read_csv(r"StockStuff\Data\fin_data.csv", usecols=["High", "Low"])
# print(data.shape)
def gan():
    BATCH_SIZE = 32
    SEED_SIZE = 50
    EPOCHS = 20
    metrics = ["High"]

    def make_generator():
        nn = models.Sequential([
            layers.Input(shape=[SEED_SIZE, ]),
            layers.Dense(len(metrics) * SEED_SIZE),
            layers.BatchNormalization(),
            layers.LeakyReLU(),

            layers.Reshape([SEED_SIZE, len(metrics), 1]),
            layers.Conv2DTranspose(32, [5, 1], [1, 1], padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),

            layers.Conv2DTranspose(64, [3, 1], [2, 1], padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),

            layers.Conv2DTranspose(1, [3, 1], [2, 1], padding='same', activation='tanh')
        ], name="generator")
        return nn

    GEN_SHAPE = make_generator()(tf.ones([1, SEED_SIZE])).shape[1:]
    print(GEN_SHAPE)

    def make_discriminator():
        nn = models.Sequential([
            layers.Input(shape=GEN_SHAPE),

            layers.Conv2D(64, [5, 1], [1, 1], padding='same'),
            layers.LeakyReLU(),
            layers.Dropout(0.2),

            layers.MaxPool2D([2, 1], strides=[2, 1]),
            layers.Conv2D(128, [3, 1], [1, 1], padding='same'),
            layers.LeakyReLU(),
            layers.Dropout(0.3),

            layers.MaxPool2D([2, 1], strides=[2, 1]),
            layers.Flatten(),
            layers.Dense(1)
        ])
        return nn

    def disc_loss(pred_on_real, pred_on_fake):
        real_loss = loss_obj(y_true=tf.ones_like(pred_on_real), y_pred=pred_on_real)
        fake_loss = loss_obj(y_true=tf.zeros_like(pred_on_fake), y_pred=pred_on_fake)
        total_loss = real_loss + fake_loss
        return total_loss

    def gen_loss(pred_on_fake):
        return loss_obj(y_true=tf.ones_like(pred_on_fake), y_pred=pred_on_fake)

    def dataset_gen():
        print("starting dataset gen")
        data = pd.read_csv(r"StockStuff\Data\fin_data.csv", usecols=metrics)
        high_val = max([max(data[col]) for col in data.columns])
        for col in data.columns:
            data[col] = data[col] / high_val
        data = data.to_numpy()
        data = data.reshape([*data.shape, 1])
        data = np.array([
            data[i * GEN_SHAPE[0]: (i + 1) * GEN_SHAPE[0]] for i in range(int(data.shape[0] / GEN_SHAPE[0]))
        ])
        data = np.array([
            data[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] for i in range(int(data.shape[0] / BATCH_SIZE))
        ])
        print(data.shape)
        print("dataset gen done")
        return data

    gen_opt, disc_opt = optimizers.Adam(), optimizers.Adam()
    loss_obj = losses.BinaryCrossentropy(from_logits=True)
    generator, discriminator = make_generator(), make_discriminator()
    print("Generator")
    generator.summary()
    print("Discriminator")
    discriminator.summary()
    prepped_data = dataset_gen()

    @tf.function
    def step(real_snapshots):
        with tf.device('/GPU:0'):
            random_seed = tf.random.normal([BATCH_SIZE, SEED_SIZE])

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                gen_snapshots = generator(random_seed, training=True)
                real_pred, fake_pred = discriminator(real_snapshots), discriminator(gen_snapshots)
                generator_loss = gen_loss(fake_pred)
                discriminator_loss = disc_loss(real_pred, fake_pred)

            gen_grad = gen_tape.gradient(generator_loss, generator.trainable_variables)
            disc_grad = disc_tape.gradient(discriminator_loss, discriminator.trainable_variables)

            gen_opt.apply_gradients(zip(gen_grad, generator.trainable_variables))
            disc_opt.apply_gradients(zip(disc_grad, discriminator.trainable_variables))

        return generator_loss

    def train(epochs):
        for epoch in range(epochs):
            start = time.time()
            mean_loss_state = []
            for batch in prepped_data:
                mean_loss_state.append(step(batch))
            print(f"Epoch: {epoch + 1}, Loss: {np.mean(mean_loss_state)}, Time:{time.time() - start}")
        generator.save(r"StockStuff\Models\data_generator")

    train(EPOCHS)


"""------------------------------------------------------------------------------------------------------------------"""

"""target: get data, simulate trading"""


def dataset_gen_2(size, xcols: (list, tuple), data_path: str):
    def make_snapshot(dx):
        output_data = [dx[i:size + i] for i in range(len(dx) - size)]
        return np.array(output_data)

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

    data_x = pd.read_csv(data_path + ".csv", usecols=xcols).to_numpy()
    data_x = clean(data_x)
    data_x = make_snapshot(data_x)

    return data_x


def candle_dataset_gen(size, cols: (list, tuple), data_path: str):
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

    def get_candles(arr: np.ndarray):
        def get_num_dir(a, b):
            return b - a

        fin_arr = []
        for i in range(arr.shape[0] - 1):
            fin_arr.append(get_num_dir(arr[i], arr[i + 1]))
        fin_arr = np.array(fin_arr)
        return fin_arr

    def make_snapshot(dx):
        output_data = [dx[i:size + i] for i in range(len(dx) - size)]
        return np.array(output_data)

    data = pd.read_csv(data_path + ".csv", usecols=cols).to_numpy()
    data = clean(data)
    data = make_snapshot(data)
    data = get_candles(np.array([elem[-1] for elem in data]))
    data = data[:-1]
    data = dim_1_check(np.array(data))
    return data


def simulator(company, model_path, model_type, capital, limit=None, interval=100, emergency=True, to_graph=True):
    model = models.load_model(model_path + "\\" + model_type)

    def find_cols():
        with open(model_path + r"\usecols.json") as file:
            col_data = json.load(file)
            return col_data["Input"], col_data["Output"]

    def find_size():
        with open(model_path + r"\size.txt", "r") as file:
            return int(file.readline())

    in_cols, out_cols = find_cols()
    size = find_size()
    data = dataset_gen_2(size, xcols=in_cols, data_path=company)
    data = data.reshape(data.shape[0], size)
    if limit is not None:
        data = data[-limit:]
    predictions = model.predict(data).flatten()
    prev_buy_price = 0
    held_quantity = 0
    transaction_points = {"Buy": [], "Sell": []}
    # model.summary()

    for i in range(predictions.shape[0]):
        # Where model thinks the price will go
        current_price = data[i][-1]
        price_at_trade = data[i][-1]
        predicted_price = predictions[i]
        # Relations: current_price is price at time of evaluation
        # Price at trade is what price the stock is actually traded at.
        # Predicted price is what the model thinks the price_at_trade will be

        # If it thinks it'll go down, then
        if predicted_price < current_price:
            if held_quantity > 0:
                if prev_buy_price < predicted_price:
                    # Sell, if all these conditions are met
                    capital += price_at_trade * held_quantity
                    held_quantity = 0
                    transaction_points["Sell"].append([i - 1, current_price])
        # If it thinks it'll go up
        elif predicted_price > current_price:
            if held_quantity == 0:
                # Buy, if not already bought
                held_quantity = math.floor(capital / price_at_trade)
                prev_buy_price = price_at_trade
                capital -= prev_buy_price * held_quantity
                transaction_points["Buy"].append([i - 1, current_price])

        # Emergency exit
        if current_price * held_quantity < prev_buy_price * held_quantity - 300 and emergency:
            capital += price_at_trade * held_quantity
            held_quantity = 0
            print("\n", i, "Emergency Exit, Capital:", capital, "\n")

        # Counter
        if i % interval == 0:
            print(f"{i}, Capital: {capital + held_quantity * current_price}, "
                  f"last_bought_price: {prev_buy_price}, "
                  f"last_bought_quantity: {held_quantity}")

    # At the end of trading
    capital += current_price * held_quantity
    print("Final:", capital)
    for key in transaction_points.keys():
        transaction_points[key] = np.array(transaction_points[key]).T

    if limit is None:
        n_data = pd.read_csv(company + ".csv", usecols=out_cols).to_numpy()[size:]
    else:
        n_data = pd.read_csv(company + ".csv", usecols=out_cols).to_numpy()[-limit:]

    if to_graph:
        plt.plot(n_data, c='b', label='Truth')
        plt.plot(predictions[:-1], c='r', label='Predictions')
        plt.scatter(transaction_points["Buy"][0], transaction_points["Buy"][1],
                    s=40, c=['#3cfc1e'] * (transaction_points["Buy"].shape[1]), label="Buy")
        plt.scatter(transaction_points['Sell'][0], transaction_points["Sell"][1],
                    s=40, c=['#bf00a9'] * (transaction_points["Sell"].shape[1]), label="Sell")
        plt.legend()
        plt.show()


def direction_eval(data_path, model_path, model_type):
    model = models.load_model(model_path + "\\" + model_type)

    def find_cols():
        with open(model_path + r"\usecols.json") as file:
            col_data = json.load(file)
            return col_data["Input"], col_data["Output"]

    def find_size():
        with open(model_path + r"\size.txt", "r") as file:
            return int(file.readline())

    def get_directions(t_arr: np.ndarray, p_arr: np.ndarray):
        def get_num_dir(a, b):
            if b - a >= 0:
                return 1
            elif b - a < 0:
                return -1

        fin_t_arr, fin_p_arr = [], []
        for i in range(t_arr.shape[0] - 1):
            fin_t_arr.append(get_num_dir(t_arr[i], t_arr[i + 1]))
            fin_p_arr.append(get_num_dir(t_arr[i], p_arr[i]))
        fin_p_arr, fin_t_arr = np.array(fin_p_arr), np.array(fin_t_arr)
        return fin_t_arr, fin_p_arr

    def stats(t_change, p_change):
        record = []
        for i in range(t_change.shape[0]):
            if t_change[i] == p_change[i]:
                record.append(1)
            else:
                record.append(0)
        print(f"Accuracy: {sum(record) / len(record)}  or  {sum(record)}/{len(record)}")

    in_cols, out_cols = find_cols()
    size = find_size()
    predict_data = dataset_gen_2(size, in_cols, data_path)
    predict_data = predict_data.reshape(predict_data.shape[0], size)
    predictions = model.predict(predict_data[:-1]).flatten()
    truth_data = np.array([elem[-1] for elem in predict_data])
    print("predict feed", predict_data.shape)
    print("truth data", truth_data.shape)
    print("predictions made", predictions.shape)

    truth_data, predictions = get_directions(truth_data, predictions)
    stats(truth_data, predictions)
    # model.summary()


def direction_eval_2(data_path, model_path, model_type):
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

    def find_cols():
        with open(model_path + r"\usecols.json") as file:
            col_data = json.load(file)
            return col_data["Input"], col_data["Output"]

    def find_size():
        with open(model_path + r"\size.txt", "r") as file:
            return int(file.readline())

    def stats(t_change, p_change):
        record = []
        for i in range(t_change.shape[0]):
            if t_change[i] == p_change[i]:
                record.append(1)
            else:
                record.append(0)
        print(f"Accuracy: {sum(record) / len(record)}  or  {sum(record)}/{len(record)}")

    in_cols, out_cols = find_cols()
    size = find_size()
    model = models.load_model(model_path + "\\" + model_type)
    predict_data = dataset_gen_2(size, in_cols, data_path)
    predict_data = predict_data.reshape(predict_data.shape[0], size)
    predictions = model.predict(predict_data[:-1])
    predictions = np.array([list(preds).index(max(preds)) for preds in predictions])
    truth_data = np.array([elem[-1] for elem in predict_data])
    truth_data = get_directions(truth_data)
    print(truth_data.shape, predictions.shape)
    stats(truth_data, predictions)


def sign_simulator(company, model_path, model, capital, limit=None, interval=100, emergency=True, to_graph=True,
                   cols=None, normalize=False, future_trade=True, shorting=False):
    def find_cols():
        with open(model_path + r"\usecols.json") as file:
            col_data = json.load(file)
            return col_data["Input"], col_data["Output"]

    def find_size():
        with open(model_path + r"\size.txt", "r") as file:
            return int(file.readline())

    if cols is None:
        in_cols, out_cols = find_cols()
    else:
        in_cols = out_cols = cols
    size = find_size()
    input_data = dataset_gen_2(size, xcols=in_cols, data_path=company)
    open_data = dataset_gen_2(size, xcols=['Open'], data_path=company)
    input_data = input_data.reshape(input_data.shape[0], size)
    if limit is not None:
        input_data = input_data[-limit:]
    if normalize:
        maximum = np.max(input_data)
        input_data /= maximum
    if future_trade:
        predictions = model.predict(input_data[:-1])
    else:
        predictions = model.predict(input_data)
    predictions = np.array([list(preds).index(max(preds)) for preds in predictions])
    if normalize:
        input_data *= maximum
    prev_buy_price = 0
    held_quantity = 0
    transaction_points = {"Buy": [], "Sell": []}
    portfolio = []
    short_positions = 0
    # model.summary()

    for i in range(predictions.shape[0]):
        # Where model thinks the price will go
        current_price = input_data[i][-1]
        if not future_trade:
            price_at_trade = open_data[i][-1]
        else:
            price_at_trade = open_data[i+1][-1]
        predicted_dir = predictions[i]

        # print(predicted_dir)
        if predicted_dir == 0:
            if held_quantity > 0:
                capital += price_at_trade * held_quantity
                held_quantity = 0
                transaction_points["Sell"].append([i - 1, current_price])
            if shorting:
                if short_positions == 0:
                    short_positions = math.floor(capital / price_at_trade)
                    capital += short_positions * price_at_trade

        else:
            if shorting:
                if short_positions > 0:
                    capital -= short_positions * price_at_trade
                    short_positions = 0
            if held_quantity == 0:
                # Buy, if not already bought
                held_quantity = math.floor(capital / price_at_trade)
                prev_buy_price = price_at_trade
                capital -= price_at_trade * held_quantity
                transaction_points["Buy"].append([i - 1, current_price])

        # Emergency exit
        if current_price * held_quantity < prev_buy_price * held_quantity - 300 and emergency:
            capital += price_at_trade * held_quantity
            held_quantity = 0
            print("\n", i, "Emergency Exit, Capital:", capital, "\n")

        # Counter
        portfolio.append(capital + held_quantity * current_price)
        if i % interval == 0:
            print(f"{i}, Capital: {capital + held_quantity * current_price}, "
                  f"last_bought_price: {prev_buy_price}, "
                  f"last_bought_quantity: {held_quantity}")

    # At the end of trading
    capital += current_price * held_quantity
    capital -= current_price * short_positions
    print("Final:", capital)
    for key in transaction_points.keys():
        transaction_points[key] = np.array(transaction_points[key]).T

    if limit is None:
        n_data = pd.read_csv(company + ".csv", usecols=out_cols).to_numpy()[size:]
    else:
        n_data = pd.read_csv(company + ".csv", usecols=out_cols).to_numpy()[-limit:]

    if to_graph:
        plt.plot(n_data, c='b', label='Truth')
        plt.plot(predictions[:-1], c='r', label='Predictions')
        plt.scatter(transaction_points["Buy"][0], transaction_points["Buy"][1],
                    s=40, c=['#3cfc1e'] * (transaction_points["Buy"].shape[1]), label="Buy")
        plt.scatter(transaction_points['Sell'][0], transaction_points["Sell"][1],
                    s=40, c=['#bf00a9'] * (transaction_points["Sell"].shape[1]), label="Sell")
        plt.legend()
        plt.show()

        plt.plot(portfolio, label='portfolio')
        plt.legend()
        plt.show()

    return capital


def sign_simulator_candle(company, model_path, model_type, capital, limit=None, interval=100, emergency=True,
                          to_graph=True, cols=None):
    model = models.load_model(model_path + "\\" + model_type)

    def find_cols():
        with open(model_path + r"\usecols.json") as file:
            col_data = json.load(file)
            return col_data["Input"], col_data["Output"]

    def find_size():
        with open(model_path + r"\size.txt", "r") as file:
            return int(file.readline())

    if cols is None:
        in_cols, out_cols = find_cols()
    else:
        in_cols = out_cols = cols
    size = find_size()
    data = candle_dataset_gen(size, cols=in_cols, data_path=company)
    print(f"\n{data.shape}\n")
    # data = data.reshape(data.shape[0], size)
    if limit is not None:
        data = data[-limit:]
    predictions = model.predict(data)
    predictions = np.array([list(preds).index(max(preds)) for preds in predictions])
    prev_buy_price = 0
    held_quantity = 0
    transaction_points = {"Buy": [], "Sell": []}
    portfolio = []
    # model.summary()

    for i in range(predictions.shape[0]):
        # Where model thinks the price will go
        current_price = data[i][-1]
        price_at_trade = data[i][-1]
        predicted_dir = predictions[i]

        # print(predicted_dir)
        if predicted_dir == 0:
            if held_quantity > 0:
                capital += price_at_trade * held_quantity
                held_quantity = 0
                transaction_points["Sell"].append([i - 1, current_price])
        else:
            if held_quantity == 0:
                # Buy, if not already bought
                held_quantity = math.floor(capital / price_at_trade)
                prev_buy_price = price_at_trade
                capital -= price_at_trade * held_quantity
                transaction_points["Buy"].append([i - 1, current_price])

        # Emergency exit
        if current_price * held_quantity < prev_buy_price * held_quantity - 300 and emergency:
            capital += price_at_trade * held_quantity
            held_quantity = 0
            print("\n", i, "Emergency Exit, Capital:", capital, "\n")

        # Counter
        portfolio.append(capital + held_quantity * current_price)
        if i % interval == 0:
            print(f"{i}, Capital: {capital + held_quantity * current_price}, "
                  f"last_bought_price: {prev_buy_price}, "
                  f"last_bought_quantity: {held_quantity}")

    # At the end of trading
    capital += current_price * held_quantity
    print("Final:", capital)
    for key in transaction_points.keys():
        transaction_points[key] = np.array(transaction_points[key]).T

    if limit is None:
        n_data = pd.read_csv(company + ".csv", usecols=out_cols).to_numpy()[size:]
    else:
        n_data = pd.read_csv(company + ".csv", usecols=out_cols).to_numpy()[-limit:]

    if to_graph:
        plt.plot(n_data, c='b', label='Truth')
        plt.plot(predictions[:-1], c='r', label='Predictions')
        plt.scatter(transaction_points["Buy"][0], transaction_points["Buy"][1],
                    s=40, c=['#3cfc1e'] * (transaction_points["Buy"].shape[1]), label="Buy")
        plt.scatter(transaction_points['Sell'][0], transaction_points["Sell"][1],
                    s=40, c=['#bf00a9'] * (transaction_points["Sell"].shape[1]), label="Sell")
        plt.legend()
        plt.show()

        plt.plot(portfolio, label='portfolio')
        plt.legend()
        plt.show()


# if __name__ == "__main__":
#
#     datasets = [file[:-4] for file in os.listdir("StockStuff/model_playground/Data") if re.findall("VAL", file)]
#     stats_record = []
#     raw_stats = [[], [], []]
#     p1, p2, p3 = "StockStuff/model_playground/Models/ensemble_test", \
#                  "StockStuff/model_playground/Models/ignore_dense", \
#              "StockStuff/model_playground/Models/ensemble_test",
#     m1, m2, m3 = models.load_model(f"{p1}/val"), models.load_model(f"{p2}/train"), models.load_model(f"{p3}/train")
#     name1, name2, name3 = "val", "best", "train"
#     for dataset in datasets:
#         print(dataset)
#         try:
#             print(name1)
#             cap_a = sign_simulator("StockStuff/model_playground/Data/" + dataset,
#                                    model_path=p1,
#                                    model=m1, capital=3000, interval=2500, emergency=False, to_graph=False)
#             raw_stats[0].append(cap_a)
#             print(
#                 "------------------------------------------------------------------------------------------------------")
#             print('\n')
#             print(name2)
#             cap_b = sign_simulator("StockStuff/model_playground/Data/" + dataset,
#                                    model_path=p2,
#                                    model=m2, capital=3000, interval=2500, emergency=False, to_graph=False)
#             raw_stats[1].append(cap_b)
#             stats_record.append((cap_a - cap_b) / cap_b)
#             print((cap_a - cap_b) / cap_b)
#             print(
#                 "------------------------------------------------------------------------------------------------------")
#             # print('\n')
#             # print(name3)
#             # cap_c = sign_simulator("StockStuff/model_playground/Data/" + dataset,
#             #                        model_path=p3,
#             #                        model=m3, capital=3000, interval=2500, emergency=False, to_graph=False)
#             # raw_stats[2].append(cap_c)
#         except:
#             continue
#
#     print(f"raw average, {name1} is better than {name2} by {sum(stats_record) / len(stats_record) * 100}, %")
#     print(
#         f"adjusted average, {name1} is better than {name2} by {sum(stats_record) - max(stats_record) - min(stats_record) / len(stats_record) * 100}, %")
#     plt.plot(raw_stats[0], label=name1)
#     plt.plot(raw_stats[1], label=name2)
#     plt.plot(raw_stats[2], label=name3)
#     plt.title("Plot of capitals")
#     plt.legend()
#     plt.show()    #  ignorer dense train is best so far
#     # But direction simple seems better than directional conv.
#     # Gotta integrate it into another ignorer

# if __name__ == "__main__":
#
#     datasets = [file[:-4] for file in os.listdir("StockStuff/model_playground/Data") if re.findall("VAL", file)]
#     stats_record = []
#     raw_stats = [[], [], []]
#     p1, p2, p3 = "StockStuff/model_playground/Models/ensemble_2", \
#              "StockStuff/model_playground/Models/ensemble_2", \
#              "StockStuff/model_playground/Models/ignore_dense"
#     m1, m2, m3 = models.load_model(f"{p1}/val"), models.load_model(f"{p2}/train"), models.load_model(f"{p3}/train")
#     name1, name2, name3 = "val", "train", "best"
#     for dataset in datasets:
#         print(dataset)
#         try:
#             print(name1)
#             cap_a = sign_simulator("StockStuff/model_playground/Data/" + dataset,
#                                    model_path=p1,
#                                    model=m1, capital=3000, interval=2500, emergency=False, to_graph=False)
#             raw_stats[0].append(cap_a)
#             print(
#                 "------------------------------------------------------------------------------------------------------")
#             print('\n')
#             print(name2)
#             cap_b = sign_simulator("StockStuff/model_playground/Data/" + dataset,
#                                    model_path=p2,
#                                    model=m2, capital=3000, interval=2500, emergency=False, to_graph=False)
#             raw_stats[1].append(cap_b)
#             stats_record.append((cap_a - cap_b) / cap_b)
#             print((cap_a - cap_b) / cap_b)
#             print(
#                 "------------------------------------------------------------------------------------------------------")
#             print('\n')
#             print(name3)
#             cap_c = sign_simulator("StockStuff/model_playground/Data/" + dataset,
#                                    model_path=p3,
#                                    model=m3, capital=3000, interval=2500, emergency=False, to_graph=False)
#             raw_stats[2].append(cap_c)
#         except:
#             continue
#
#     print(f"raw average, {name1} is better than {name2} by {sum(stats_record) / len(stats_record) * 100}, %")
#     print(
#         f"adjusted average, {name1} is better than {name2} by {sum(stats_record) - max(stats_record) - min(stats_record) / len(stats_record) * 100}, %")
#     plt.plot(raw_stats[0], label=name1)
#     plt.plot(raw_stats[1], label=name2)
#     plt.plot(raw_stats[2], label=name3)
#     plt.title("Plot of capitals")
#     plt.legend()
#     plt.show()

"""Market order: Very small variation between eval_price and traded_price"""
"""To do: Work on trading mechanic, add plotting options"""
"""Report on dir_only
"""

""" What sets dir-conv apart
weights are all between 0.075 and -0.075"""

"""ALso, maybe add an emergency slide of larger diff"""

"""Best: ignore dense train. Nope, now its ensemble"""