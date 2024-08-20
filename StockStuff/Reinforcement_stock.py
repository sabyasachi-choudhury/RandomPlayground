"""Welcome to reinforcements!"""
import random

print("start")
from sklearn import linear_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from tensorflow.keras import models
import tensorflow as tf

print("imported")


def make_snapshots(arr: np.ndarray, size: int):
    length = arr.shape[0]
    snaps = []
    labels = []
    for i in range(int(length - size - 1)):
        snaps.append(arr[i:i + size])
        labels.append(arr[i + size + 1])
    return np.array(snaps, dtype=float), np.array(labels, dtype=float)


# print("making")
# dx, dy = make_snapshots(data, 4)
# print(dx.shape, dy.shape)
#
# model.fit(X=dx, y=dy)
# preds = np.concatenate(
#     [
#      np.zeros([4]),
#      model.predict(dx)
#      ]
# )
# plt.plot(preds, label="preds")
# plt.plot(data, label="truth")
# plt.legend()
# plt.show()

def simulator(data, label_data, model, capital, size, limit=None, interval=1000, emergency=True, to_graph=True):
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
        n_data = label_data[size:]
    else:
        n_data = label_data[-limit:]

    if to_graph:
        plt.plot(n_data, c='b', label='Truth')
        plt.plot(predictions[:-1], c='r', label='Predictions')
        plt.scatter(transaction_points["Buy"][0], transaction_points["Buy"][1],
                    s=40, c=['#3cfc1e'] * (transaction_points["Buy"].shape[1]), label="Buy")
        plt.scatter(transaction_points['Sell'][0], transaction_points["Sell"][1],
                    s=40, c=['#bf00a9'] * (transaction_points["Sell"].shape[1]), label="Sell")
        plt.legend()
        plt.show()
    return capital


def moving_average_model(data_x: np.ndarray, data_y: np.ndarray, period, weights=None, capital=3000):
    if weights is None:
        weights = np.array([1]*period)

    held_quantity = 0

    for i in range(data_x.shape[0]-1-period):
        snap = data_x[i:i+period].flatten()
        prediction = np.sum(snap*weights)/sum(weights)
        current = snap[-1]
        # p = (1000 + random.randint(-10, 10))/1000
        at_transaction = data_y[i+period]
        if prediction > current:
            if held_quantity == 0:
                held_quantity = int(capital/current)
                capital -= at_transaction * held_quantity
        elif prediction < current:
            if held_quantity > 0:
                capital += held_quantity * at_transaction
                held_quantity = 0

    capital += current * held_quantity
    print(f"capital: {capital}")
    return capital

def moving_regression_model(data_x: np.ndarray, data_y: np.ndarray, period: int, batches: int, capital=3000):
    held_quantity = 0
    for i in range(data_x.shape[0]-batches-period):
        dx = data_x[i: i+batches+period-1]
        dx = [dx[j:j+period] for j in range(batches)]
        dy = [data_y[i+j+period] for j in range(batches)]
        # print(dx)
        # print(dy)
        model = linear_model.LinearRegression()
        model.fit(dx, dy)

        pred_feed = [data_x[i+batches: i+batches+period]]
        # print(pred_feed)
        current = dy[-1]
        prediction = model.predict(pred_feed)
        at_trade = data_y[i+period+batches]
        # print(pred_feed, at_trade)
        # print('\n')
        if prediction > current:
            if held_quantity == 0:
                held_quantity = int(capital/current)
                capital -= at_trade * held_quantity
        elif prediction < current:
            if held_quantity > 0:
                capital += held_quantity * at_trade
                held_quantity = 0
    capital += current * held_quantity
    print(f"capital: {capital}")
    return capital


datasets = [file[:-4] for file in os.listdir("StockStuff/model_playground/Data")]
raw_stats = []

# for dataset in datasets:
#     print("\n")
#     print(dataset)
#     # weights = tf.nn.softmax([1., 1., 2., 3., 4.])
#     # weights = [1, 4, 9, 16, 25]
#     weights = None
#     # res = moving_average_model(data_x=pd.read_csv("StockStuff/model_playground/Data/" + dataset + ".csv", usecols=["High"]).to_numpy(),
#     #                            data_y=pd.read_csv("StockStuff/model_playground/Data/" + dataset + ".csv", usecols=["High"]).to_numpy(),
#     #                            period=5,
#     #                            weights=weights)
#     res = moving_regression_model(
#         data_x=pd.read_csv("StockStuff/model_playground/Data/" + dataset + ".csv", usecols=["High"]).to_numpy().flatten(),
#         data_y=pd.read_csv("StockStuff/model_playground/Data/" + dataset + ".csv", usecols=["High"]).to_numpy().flatten(),
#         period=4,
#         batches=50)
#
#     raw_stats.append(res)

# print(
#     (sum(raw_stats)-max(raw_stats))/(len(raw_stats)-1)
# )
# plt.plot(raw_stats)
# plt.show()


from StockAnalysis2 import sign_simulator

if __name__ == "__main__":

    datasets = [file[:-4] for file in os.listdir("StockStuff/model_playground/Data")]
    datasets = [ds for ds in datasets if ds[-3:] == "VAL"]
    # datasets = [file[:-4] for file in os.listdir("StockStuff/YahooData")]
    stats_record = []
    raw_stats = [[], []]
    p1, p2 = "StockStuff/model_playground/Models/close_to_open", \
             "StockStuff/model_playground/Models/ignore_dense",
    m1, m2 = models.load_model(p1 + "/val"), models.load_model(f"{p2}/train")
    name1, name2, name3 = "val", "best", "best"
    for dataset in datasets:
        print(dataset)
        # print('\n')
        # print(name1)
        # cap_a = sign_simulator("StockStuff/model_playground/Data/" + dataset,
        #                        model_path=p1,
        #                        model=m1, capital=3000, interval=2500, emergency=False, to_graph=False)
        # raw_stats[0].append(cap_a)
        "------------------------------------------------------------------------------------------------------"
        print('\n')
        print(name2)
        cap_b = sign_simulator("StockStuff/model_playground/Data/" + dataset,
                               model_path=p2,
                               model=m2, capital=3000, interval=1000, emergency=False, to_graph=True, future_trade=True, shorting=True)
        raw_stats[1].append(cap_b)
        # stats_record.append((cap_a - cap_b) / cap_b)
        # print((cap_a - cap_b) / cap_b)

    # print(f"raw average, {name1} is better than {name2} by {sum(stats_record) / len(stats_record) * 100}, %")
    # print(
    #     f"adjusted average, {name1} is better than {name2} by {sum(stats_record) - max(stats_record) - min(stats_record) / len(stats_record) * 100}, %")
    # plt.plot(raw_stats[0], label=name1)
    print(f"Average Capital {sum(raw_stats[1])/len(raw_stats[1])}")
    print(f"Adjusted Capital {(sum(raw_stats[1]) - max(raw_stats[1]))/(len(raw_stats[1])-1)}")
    plt.plot(raw_stats[1], label=name2)
    plt.title("Plot of capitals")
    plt.legend()
    plt.show()