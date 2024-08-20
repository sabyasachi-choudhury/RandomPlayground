import backtesting as bt
import numpy as np
from tensorflow.keras import models
import pandas as pd
import yfinance as yf
import os
print("Imported")

# ticker = yf.Ticker('AAPL')
# data = ticker.history(period="60d", interval="15m", actions=True)
# print(data)
# data.to_csv("StockStuff/YahooData/AAPL.csv")

def rolling_array(arr, size):
    arr = [arr[i:i+size] for i in range(len(arr)-size)]
    return np.array(arr)

def predict(values, size):
    predictions = model.predict(rolling_array(values, size))
    predictions = np.array([0 if pair[0] > pair[1] else 1 for pair in predictions])
    predictions = np.concatenate((
        np.array([np.nan]*size),
        predictions
    ))
    print(predictions)
    return np.array(predictions)

class Strategy(bt.Strategy):
    def init(self):
        self.prediction = self.I(predict, self.data.Low, 20)

    def next(self):
        # If decrease
        if self.prediction[-1] == 0:
            self.sell()
        else:
            self.buy()

symbol = 'PFE'
model = models.load_model("StockStuff/model_playground/Models/ignore_dense/train")
present_datasets = [file[:-4] for file in os.listdir("StockStuff/YahooData")]
print(present_datasets)
if symbol not in present_datasets:
    data = yf.Ticker(symbol).history(period='60d', interval='15m', actions=True)
    data.to_csv(f"StockStuff/YahooData/{symbol}.csv")
else:
    data = pd.read_csv(f"StockStuff/YahooData/{symbol}.csv", usecols=['Datetime', 'Open', 'High', 'Low', 'Close'])

engine = bt.Backtest(data=data, strategy=Strategy, cash=3000, commission=0.005)
stats = engine.run()
engine.plot()
print(stats)