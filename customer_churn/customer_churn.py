import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.losses import BinaryCrossentropy

print(len(tf.config.list_physical_devices()))

"""churned- yes:left, no:still there"""
def g_convert(g):
    g.replace('Female', '1')
    g.replace('Male', '0')

data = pd.read_csv(
    r"C:\Users\Sabyasachi\PycharmProjects\TestGroundTwo\customer_churn\WA_Fn-UseC_-Telco-Customer-Churn.csv",
    usecols=list(range(1, 21))
    )

churns = pd.read_csv(
    r"C:\Users\Sabyasachi\PycharmProjects\TestGroundTwo\customer_churn\WA_Fn-UseC_-Telco-Customer-Churn.csv",
    usecols=['Churn']
    )

def count(name, details=False):
    counts = {}
    for x in data[name]:
        if x not in counts:
            counts[x] = 1
        else:
            counts[x] += 1
    if details:
        print(name, counts)
    return counts

"""replace stuff"""
data['Churn'] = churns
data = data.replace('Male', 0)
data = data.replace('Female', 1)

data = data.replace('Yes', 1)
data = data.replace('No', 0)
data = data.replace('No phone service', 0)

data = data.replace('DSL', 1)
data = data.replace('Fiber optic', 2)

data = data.replace('No internet service', 0)

data = data.replace('Month-to-month', 0)
data = data.replace('One year', 1)
data = data.replace('Two year', 2)

data = data.replace('Electronic check', 0)
data = data.replace('Mailed check', 1)
data = data.replace('Bank transfer (automatic)', 2)
data = data.replace('Credit card (automatic)', 3)

data = data.replace(' ', 0)

all_counts = {name: count(name) for name in data.columns}

data_dict = {name: data[name].values for name in data.columns}

for d in data_dict:
    data_dict[d] = data_dict[d].astype(np.float16)
    print(data_dict[d].dtype, data_dict[d], d)


def churn_indexer(details=False):
    exceptions = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']
    # indexes = {'Churned': {col: {key: 0 for key in count(col).keys()}
    # for col in data.columns if col not in exceptions},
    #            'Not Churned': {col: {key: 0 for key in count(col).keys()}
    #            for col in data.columns if col not in exceptions}}
    indexes = {'Churned': [], 'Not Churned': []}
    churn_positions = {'Churned': [i for i in range(len(data_dict['Churn'])) if data_dict['Churn'][i] == 1],
                       'Not Churned': [i for i in range(len(data_dict['Churn'])) if data_dict['Churn'][i] == 0]}

    for key in churn_positions.keys():
        for index in churn_positions[key]:
            indexes[key].append([data_dict[col][index] for col in data_dict.keys() if col != 'Churn'])

    print('\n')
    if details:
        for key in indexes:
            if key not in exceptions:
                print(key, indexes[key])
    return indexes


def dataset_gen():
    indexes = churn_indexer()
    data_x = indexes['Churned'].copy()
    data_x.extend(indexes['Not Churned'])
    data_y = [1] * len(indexes['Churned'])
    data_y.extend([0] * len(indexes['Not Churned']))

    rand_indices = np.random.permutation(len(data_y))

    data_x, data_y = np.array(data_x), np.array(data_y)
    tX, tY = data_x[rand_indices[:-100]], data_y[rand_indices[:-100]]
    t_X, t_Y = data_x[rand_indices[-100::]], data_y[rand_indices[-100::]]

    return (tX, tY), (t_X, t_Y)


(train_x, train_y), (test_x, test_y) = dataset_gen()
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

"""DNN STUFF"""
with tf.device('/GPU:0'):
    model = models.Sequential([
        layers.Dense(19, activation='relu', input_shape=[19, ]),
        layers.Dense(30, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss=BinaryCrossentropy(), metrics=['accuracy'])
    print("Compiled. Starting training.")
    model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=50)