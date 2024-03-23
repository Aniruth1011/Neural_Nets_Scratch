import numpy as np
#from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd

def get_data():
    boston = pd.read_csv('BostonHousing.csv')

    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    x = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    y = raw_df.values[1::2, 2]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0)
    x_test = (x_test - x_train.mean(axis=0)) / x_train.std(axis=0)

    x_train = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
    x_test = np.hstack((np.ones((x_test.shape[0], 1)), x_test))

    return x_train , y_train , x_test , y_test 
