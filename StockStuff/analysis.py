import math
import random
import time
import re
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from tensorflow.keras import models
import tensorflow as tf
import json
import os

"""Remember: Dense 1 was the best
arch = dense 30, dense20, dense1"""


def high_mean_low(company):
    data = pd.read_csv(r"C:\Users\Sabyasachi\PycharmProjects\TestGroundTwo\StockStuff\Data" + "\\" + company + ".csv")

    mean_snip = data["Mean"]
    high_snip = data["High"]
    low_snip = data["Low"]

    plt.plot(mean_snip, label="mean")
    plt.plot(high_snip, label="high")
    plt.plot(low_snip, label="low")
    plt.legend()
    plt.show()


def comparison(model_name, csv_name, graph=True, record=True, data_path_base=r"StockStuff\Data", abs_val=True,
               model_path_base=r"StockStuff\Models", spec_size=False):
    model_path = model_path_base + "\\" + model_name
    available_models = [f for f in os.listdir(model_path) if os.path.isdir(model_path + "\\" + f)]

    def find_cols():
        with open(model_path + r"\usecols.json") as file:
            col_data = json.load(file)
            return col_data["Input"], col_data["Output"]

    def dim_1_check(arr: np.ndarray):
        if arr.shape[-1] == 1:
            return arr.reshape([elem for elem in arr.shape[:-1]])
        else:
            return arr

    def snapshot(size, d):
        output = [d[i:i + size] for i in range(len(d) - size + 1)]
        return np.array(output)

    def find_size():
        with open(model_path + r"\size.txt", "r") as file:
            return int(file.readline())

    def loss_avg(p, t):
        if abs_val:
            loss_list = abs(p-t)
        else:
            loss_list = p-t
        return sum(loss_list) / len(loss_list), max(loss_list), min(loss_list)

    def loss_display(p, t):
        stat_rec = {}
        for i in range(len(OUT_COLS)):
            print(OUT_COLS[i])
            loss_mean, loss_max, loss_min = loss_avg(p[i][SIZE:], t[i][SIZE:])
            print("mean error:", loss_mean)
            print("max error:", loss_max)
            print("min error:", loss_min)
            stat_rec[OUT_COLS[i]] = {"mean_loss": loss_mean, "max_loss": loss_max, "min_loss": loss_min}
        if record:
            record_stats(stats=stat_rec)

    def record_stats(stats):
        with open(model_path + "\\" + m_path + r"\stats.json", "w") as file:
            json.dump(stats, file, indent=2)

    def buffering(arr: np.ndarray):
        if len(arr.shape) == 1:
            if not spec_size:
                buffer = np.zeros((SIZE - 1,))
                return np.concatenate((buffer, arr))
            else:
                buffer = np.zeros((15))
                return np.concatenate((buffer, arr))
        else:
            buffer = np.zeros([SIZE - 1] + list(arr.shape[1:]))
            return np.concatenate((buffer, arr))

    def plot_prep(pred, do_buffer=False):
        preds = pred.copy()
        output = []
        if do_buffer:
            preds = buffering(preds)
        if len(preds.shape) != 1:
            for aspect in range(preds.shape[1]):
                output.append(np.array([group[aspect] for group in preds]))
        else:
            output.append(preds)
        return output

    IN_COLS, OUT_COLS = find_cols()
    data = dim_1_check(pd.read_csv(data_path_base + "\\" + csv_name + ".csv", usecols=IN_COLS).to_numpy())
    label_data = dim_1_check(pd.read_csv(data_path_base + "\\" + csv_name + ".csv", usecols=OUT_COLS).to_numpy())
    label_data = np.array(plot_prep(label_data))

    if not spec_size:
        SIZE = find_size()
        feed = dim_1_check(snapshot(SIZE, data))
    else:
        SIZE = 1
        feed = data

    for m_path in available_models:
        model = models.load_model(model_path + "\\" + m_path)
        model.summary()

        predictions = np.array(plot_prep(dim_1_check(model.predict(feed)), do_buffer=True))
        print(predictions.shape, label_data.shape)
        print(m_path)

        if graph:
            for i in range(len(OUT_COLS)):
                plt.plot(predictions[i], label="predictions")
                plt.plot(label_data[i], label="truth")
                plt.title(model_name + " : " + OUT_COLS[i])
                plt.legend()
                plt.show()

        loss_display(predictions, label_data)


def complete_pred(model_name, dataset):
    model_path = r"Stockstuff\Models" + "\\" + model_name
    available_models = [f for f in os.listdir(model_path) if os.path.isdir(model_path + "\\" + f)]

    def find_cols():
        with open(model_path + r"\usecols.json") as file:
            col_data = json.load(file)
            return col_data["Input"], col_data["Output"]

    def find_size():
        with open(model_path + r"\size.txt", "r") as file:
            return int(file.readline())

    def dim_1_check(arr: np.ndarray):
        if arr.shape[-1] == 1:
            return arr.reshape([elem for elem in arr.shape[:-1]])
        else:
            return arr

    def buffering(arr: np.ndarray):
        if len(arr.shape) == 1:
            buffer = np.zeros((SIZE - 1,))
            return np.concatenate((buffer, arr))
        else:
            buffer = np.zeroes([SIZE - 1] + arr.shape[1:])
            return np.concatenate((buffer, arr))

    def plot_prep(pred, do_buffer=False):
        preds = pred.copy()
        output = []
        if do_buffer:
            preds = buffering(preds)
        if len(preds.shape) != 1:
            for aspect in range(preds.shape[1]):
                output.append(np.array([group[aspect] for group in preds]))
        else:
            output.append(preds)
        return output

    IN_COLS, OUT_COLS = find_cols()
    if IN_COLS == OUT_COLS:

        SIZE = find_size()
        label_data = dim_1_check(dataset)
        num_data_points = len(label_data)
        label_data = np.array(plot_prep(label_data))

        for m_path in available_models:
            model = models.load_model(model_path + "\\" + m_path)
            model.summary()
            predictions = dim_1_check(dataset)[:SIZE]
            print(predictions.shape)

            start = time.time()
            for i in range(num_data_points):
                with tf.device('/GPU:0'):
                    new_pred_feed = np.array([predictions[-1 * SIZE:]])
                    new_prediction = dim_1_check(model.predict(new_pred_feed))
                    predictions = np.concatenate((predictions, new_prediction), axis=0)
            print(time.time() - start)

            print(predictions.shape)
            predictions = plot_prep(dim_1_check(predictions))

            for i in range(len(OUT_COLS)):
                plt.plot(predictions[i], label="predictions")
                plt.plot(label_data[i], label="truth")
                plt.title(OUT_COLS[i])
                plt.legend()
                plt.show()


def error_check():
    errs = {s: [] for s in ["Data 1", "Data 2", "Data 3", "Data 4"]}
    for s in ["Data 1", "Data 2", "Data 3", "Data 4"]:
        data = pd.read_csv(r"StockStuff\FTSE" + "\\" + s + ".csv", usecols=['Low']).to_numpy().flatten()
        for i, high in enumerate(data):
            if i > 1:
                if high - data[i - 1] > 3000:
                    errs[s].append(i)

    print(errs)
    with open(r"StockStuff\FTSE\low_errors.json", "w") as file:
        json.dump(errs, file, indent=2)


def final_data_make():
    def create_mean(df):
        mean_col = [(df["High"][i] + df["Low"][i]) / 2 for i in range(len(df["High"]))]
        return mean_col

    base_df = pd.read_csv(r"StockStuff\FTSE\Data 1.csv", usecols=['Date', 'High', 'Low'])
    error_indices = [183884, 189013]
    for col in ["High", "Low"]:
        for ind in error_indices:
            base_df[col][ind] = (base_df[col][ind - 1] + base_df[col][ind + 1]) / 2
    base_df['Mean'] = create_mean(base_df)
    print(base_df.shape)

    for name in ["Data 2", "Data 3", "Data 4"]:
        print(name)
        data = pd.read_csv(r"StockStuff\FTSE" + "\\" + name + ".csv", usecols=['Date', 'High', 'Low'])
        data['Mean'] = create_mean(data)
        base_df = base_df.append(data)

    print(base_df.head(), base_df.shape)

    # base_df.to_csv(r"StockStuff\Data\fin_data.csv")


def final_data_make_2(df):
    def to_daily():
        with open(r"StockStuff\ExtraData\dates.json") as file:
            dates = json.load(file)["dates"]
        out_arr = pd.DataFrame({"High": [], "Low": [], "Mean": []})
        bookmark = 0
        print("starting loop")
        start = time.time()
        for date in dates:
            start2 = time.time()
            date_data = {"High": [], "Low": [], "All": []}
            for i in range(bookmark, len(df["Date"])):
                if df.iloc[i]["Date"] == date:
                    date_data["High"].append(df["High"][i])
                    date_data["Low"].append(df["Low"][i])
                    date_data["All"].extend([df["High"][i], df["Low"][i]])
                if i < df["Date"].shape[0] - 1:
                    if df["Date"][i + 1] != df["Date"][i]:
                        bookmark = i + 1
                        break

            out_arr = out_arr.append(
                pd.DataFrame(
                    {"High": [max(date_data["High"])],
                     "Low": [min(date_data["Low"])],
                     "Mean": [np.mean(date_data["All"])]}
                )
            )
            print(date, time.time() - start2)
        print(time.time() - start)
        print(out_arr, out_arr.shape)

        return out_arr

    new_data = to_daily()
    new_data.to_csv(r"StockStuff\ExtraData\FTSEDaily.csv")


def limit_pred(limit, data_path, model_path, model_type, start_ind=None):
    def find_cols():
        with open(model_path + r"\usecols.json") as file:
            col_data = json.load(file)
            return col_data["Input"], col_data["Output"]

    def find_size():
        with open(model_path + r"\size.txt", "r") as file:
            return int(file.readline())

    def dim_1_check(arr: np.ndarray):
        if arr.shape[-1] == 1:
            return arr.reshape([elem for elem in arr.shape[:-1]])
        else:
            return arr

    model = models.load_model(model_path + "/" + model_type)
    in_cols, out_cols = find_cols()
    data = dim_1_check(pd.read_csv(data_path, usecols=in_cols).to_numpy())
    size = find_size()

    if start_ind is None:
        start_ind = random.randint(0, data.shape[0])
    data_slice = data[start_ind - size:start_ind]
    label_slice = data[start_ind:start_ind + limit].T
    for i in range(limit):
        new_point = model.predict(np.array([data_slice[i:i + size]]))
        data_slice = np.concatenate([data_slice, new_point[0]])
    data_slice = data_slice[size:].T
    print(data_slice)
    print(label_slice)

    if len(out_cols) != 1:
        for i, col in enumerate(out_cols):
            plt.plot(data_slice[i], label="Prediction")
            plt.plot(label_slice[i], label="Truth")
            plt.legend()
            plt.show()
    else:
        plt.plot(data_slice, label="Prediction")
        plt.plot(label_slice, label="Truth")
        plt.legend()
        plt.show()


def spec_stats(model_path, model_type, csv_path, period):
    def find_cols():
        with open(model_path + r"\usecols.json") as file:
            col_data = json.load(file)
            return col_data["Input"], col_data["Output"]

    def find_size():
        with open(model_path + r"\size.txt", "r") as file:
            return int(file.readline())

    def snapshots(d: np.ndarray, p: int, s: int):
        num_preds = math.floor((d.shape[0] - s) / p)
        snaps = np.array([d[i * p: i * p + s] for i in range(num_preds)])
        return snaps

    def dim_1_check(arr: np.ndarray):
        if arr.shape[-1] == 1:
            return arr.reshape([elem for elem in arr.shape[:-1]])
        else:
            return arr

    def metrics(p: np.ndarray, t: np.ndarray):
        loss_arr = abs(p-t)
        if len(in_cols) == 1:
            print(in_cols[0])
            print("Mean Loss:", np.average(loss_arr))
            print("Max Loss:", np.max(loss_arr))
            print("Min Loss:", np.min(loss_arr))
        else:
            for i, col in enumerate(in_cols):
                print(col)
                print("Mean Loss:", np.average(loss_arr[:, i]))
                print("Max Loss:", np.max(loss_arr[:, i]))
                print("Min Loss:", np.min(loss_arr[:, i]))

    model = models.load_model(model_path + "\\" + model_type)
    print(model.summary())
    in_cols, out_cols = find_cols()
    if in_cols != out_cols:
        raise Exception("NoRecursivePredictionPossible")
    size = find_size()
    input_data = pd.read_csv(csv_path, usecols=in_cols).to_numpy()
    input_data = dim_1_check(snapshots(d=input_data, p=period, s=size))
    print("Input_shape_1:", input_data.shape)
    label_data = dim_1_check(pd.read_csv(csv_path, usecols=out_cols).to_numpy()[size:])
    predictions = model.predict(input_data)
    predictions = predictions.reshape(predictions.shape[0], 1, predictions.shape[-1])
    print("Predictions_shape_1:", predictions.shape)

    for i in range(2, period+1):
        input_data = np.concatenate([
            input_data[:, 1:],
            predictions[:, -1]
        ], axis=-1)
        print("Input_shape_" + str(i), input_data.shape)
        new_predictions = model.predict(input_data)
        new_predictions = new_predictions.reshape(new_predictions.shape[0], 1, new_predictions.shape[-1])
        predictions = np.concatenate([predictions, new_predictions], axis=1)
        print("Prediction shape " + str(i), predictions.shape)

    if predictions.shape[-1] != 1:
        predictions = predictions.reshape([predictions.shape[0]*predictions.shape[1], predictions.shape[-1]]).T
        label_data = label_data[:predictions.shape[0]].T
        print("FinalLabelShape:", label_data.shape)
        print("FinalPredictionShape:", predictions.shape)
    else:
        predictions = predictions.flatten()
        label_data = label_data.flatten()[:predictions.shape[0]]
        print("FinalLabelShape:", label_data.shape)
        print("FinalPredictionShape:", predictions.shape)
        plt.plot(predictions, label="Predictions")
        plt.plot(label_data, label="LabelData")
        plt.legend()
        plt.show()
        metrics(predictions, label_data)


# comparison(model_name='low_to_low_2', model_path_base="StockStuff/model_playground/Models",
#            data_path_base="StockStuff/model_playground/Data", csv_name="LGVN_VAL", record=False)
# limit_pred(50, r"StockStuff/model_playground/Data/PFE_VAL.csv", r"StockStuff/model_playground/Models/low_to_low_2", "val", 40000)
# limit_pred(200, "StockStuff/model_playground/Data/PFE_VAL.csv", "StockStuff/model_playground/Models/high_to_high", "val")

# spec_stats(model_path=r"StockStuff\model_playground\Models\low_to_low_2", model_type="val",
#            csv_path="StockStuff/model_playground/Data/PFE_VAL.csv", period=3)


"""Look for much longer patterns. Try MAP"""