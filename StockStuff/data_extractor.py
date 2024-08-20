import functools
import pandas as pd
import yfinance as yf
import json
import numpy as np
import math
# import tensorflow as tf
# from tensorflow.keras.models import load_model
import time

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


def process(code):
    comp = yf.Ticker(code)
    data = comp.history("max")
    del data["Stock Splits"]
    del data["Dividends"]
    mean_col = [(data["High"][i] + data["Low"][i]) / 2 for i in range(len(data["High"]))]
    data["Mean"] = mean_col
    # print(data.head())
    return data


def model_predict_data():
    models = ['high_output_ftse_min', 'ftse_min_conv_high', 'multi_output_ftse_min']

    def timeit(func):
        @functools.wraps(func)
        def modified_func(purpose, *args, **kwargs):
            start = time.time()
            output = func(*args, **kwargs)
            print("Time Taken for " + purpose, time.time() - start)
            return output
        return modified_func

    def model_path(model_name):
        return r"StockStuff\Models" + "\\" + model_name + "\\" + "val"

    def find_size(model_name):
        with open(r"StockStuff\Models" + "\\" + model_name + r"\size.txt", "r") as file:
            return int(file.readline())

    def find_cols(model_name):
        with open(r"StockStuff\Models" + "\\" + model_name + r"\usecols.json") as file:
            col_data = json.load(file)
            return col_data["Input"], col_data["Output"]

    @timeit
    def snapshot(size, d):
        output = [d[i:i + size] for i in range(len(d) - size + 1)]
        return np.array(output)

    def dim_1_check(arr: np.ndarray):
        if arr.shape[-1] == 1:
            return arr.reshape([elem for elem in arr.shape[:-1]])
        else:
            return arr

    @timeit
    def np_to_df(arr: np.ndarray, col_names: (list, np.ndarray)):
        t_shape = [arr.shape[1], arr.shape[0]]
        output_arr = np.array([[sample[col_ind] for sample in arr] for col_ind in range(t_shape[0])])
        output_df = pd.DataFrame({col_name: output_arr[i] for i, col_name in enumerate(col_names)})
        return output_df

    @timeit
    def get_predictions(f, nn):
        return nn.predict(f)

    for name in models:
        print(name)
        IN_COLS, OUT_COLS = find_cols(name)
        data = pd.read_csv(r"StockStuff\Data\fin_data.csv", usecols=IN_COLS).to_numpy()
        SIZE = find_size(name)
        model = load_model(model_path(name))

        feed = snapshot(size=SIZE, d=data, purpose="dataset gen")
        predictions = get_predictions(f=feed, nn=model, purpose="making predictions")
        to_save = np_to_df(arr=predictions, col_names=OUT_COLS, purpose="make dataframe")
        to_save.to_csv(r"StockStuff\ExtraData" + "\\" + name + "_data")


def all_model_data():
    names = ['high_output_ftse_min_data', 'multi_output_ftse_min_data']
    base_data = pd.read_csv(r'StockStuff\ExtraData\ftse_min_conv_high_data', usecols=['High'])
    base_data = base_data.rename(columns={"High": "ftse_min_conv_high"})
    for n in names:
        n_data = pd.read_csv(r"StockStuff\ExtraData" + "\\" + n, usecols=["High"]).to_numpy()[5:]
        n_data = pd.DataFrame(n_data, columns=[n])
        base_data[n] = n_data

    true_data = pd.read_csv(r"StockStuff\Data\fin_data.csv", usecols=["High"]).to_numpy()[14:]
    true_data = pd.DataFrame(true_data, columns=["High"])
    base_data["High"] = true_data
    print(base_data, base_data.shape)
    base_data.to_csv(r"StockStuff\Data\all_model_data.csv")


def convert_to_period(period, csv_path, save, cols):
    raw_data = pd.read_csv(csv_path, usecols=cols).to_numpy().T
    fin_data = {"Low": [], "High": [], "Mean": []}
    for i in range(math.floor(raw_data.shape[1]/period)):
        temp_snap = raw_data[:, i*period:(i+1)*period]
        fin_data["Low"].append(min(temp_snap[cols.index("Low")]))
        fin_data["High"].append(max(temp_snap[cols.index("High")]))
        fin_data["Mean"].append(np.mean(temp_snap[cols.index("Mean")]))
        if i%2000 == 0:
            print(i)
    fin_data = pd.DataFrame(fin_data)
    print(fin_data.shape)
    fin_data.to_csv(save)


# all_model_data()
# d1 = pd.read_csv(r"StockStuff\Data\all_model_data.csv", usecols=['high_output_ftse_min_data', 'ftse_min_conv_high'])
# print(d1.head(3))
#
# d2 = pd.read_csv(r"StockStuff\ExtraData\high_output_ftse_min_data", usecols=["High"])
# print(d2.head(20))
#
# d3 = pd.read_csv(r"StockStuff\ExtraData\ftse_min_conv_high_data", usecols=["High"])
# print(d3.head(20))
# convert_to_period(15, "StockStuff/Data/fin_data.csv", "StockStuff/ExtraData/FTSE_15min_train.csv", ["High", "Low", "Mean"])