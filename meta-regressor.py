import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn import metrics

def linear(csv_path, target):

    df = pd.read_csv(csv_path)
    del df['Unnamed: 0'] # a small workaround regarding csv exporting issues
    del df['db']

    df = df.fillna(0)
    df = df.replace(np.inf, 100)
    df = df.replace(-np.inf, -100)

    y = df[target]
    del df[target]
    x = df

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

    linear = LinearRegression()
    linear.fit(x_train, y_train)
    y_pred = linear.predict(x_test)

    acc = metrics.mean_absolute_error(y_pred, y_test)

    print(pd.DataFrame({'predicted': y_pred, 'actual': y_test}))

    return acc

# database abalone

meta_path = './results/meta-knowledge.csv'
cart_target = 'cart_accuracy'
naive_target = 'naive_accuracy'
neural_target = 'neural_accuracy'

meta_cart_acc = linear(meta_path, cart_target)
meta_naive_acc = linear(meta_path, naive_target)
meta_neural_acc = linear(meta_path, neural_target)

table = {
    'learner': ['cart', 'naive', 'neural'],
    'score': [meta_cart_acc, meta_naive_acc, meta_neural_acc]
}

columns = ['learner', 'score']

df = pd.DataFrame(table, columns=columns)

df.to_csv('./results/meta-model.csv')
