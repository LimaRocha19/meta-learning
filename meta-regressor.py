import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# from irlmaths import outliers
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from scipy import stats

def outliers(df):

    outliers = 0
    threshold = 3

    for index, row in df.iterrows():
        for column in df.columns:
            z = (row[column] - df[column].mean()) / df[column].std()
            if np.abs(z) > threshold:
                outliers = outliers + 1
                break

    return outliers

def knn(df, target, k=3):

    if 'Unnamed: 0' in df.columns:
        del df['Unnamed: 0'] # a small workaround regarding csv exporting issues

    # metadata extraction for the new database

    df_examples = np.log2(df.count()[0])
    df_attributes = np.log2(len(df.columns))
    df_discrete_ratio = len(df.select_dtypes(include=['int64']).columns) / len(df.columns)

    df_classes = len(df[target].value_counts())
    df_entropy = stats.entropy(df[target], base=2)

    df_outliers = outliers(df) / df.count()[0]

    entropies = []

    for column in df.select_dtypes(include=['int64']).columns:
        if not(np.isnan(stats.entropy(df[column], base=2))):
            entropies.append(stats.entropy(df[column], base=2))

    df_mean_entropy = np.mean(entropies)

    for column in df.select_dtypes(include=['int64']).columns:
        del df[column]

    df_mean_correlation = np.abs(df.corr()).mean().mean()

    df_mean_skew = df.skew().mean()

    df_mean_kurtosis = df.kurtosis().mean()

    table = {
        'examples': df_examples,
        'attributes': df_attributes,
        'discrete_ratio': df_discrete_ratio,
        'mean_entropy': df_mean_entropy,
        'mean_correlation': df_mean_correlation,
        'mean_skew': df_mean_skew,
        'mean_kurtosis': df_mean_kurtosis,
        'outliers': df_outliers,
        'classes': df_classes,
        'entropy': df_entropy
    }

    columns = ['examples', 'attributes', 'discrete_ratio', 'mean_entropy', 'mean_correlation', 'mean_skew', 'mean_kurtosis', 'outliers', 'classes', 'entropy']

    meta_df = pd.DataFrame(table, columns=columns, index=[0])

    print(meta_df)

    meta_df = meta_df.fillna(0)
    meta_df = meta_df.replace(np.inf, 100)
    meta_df = meta_df.replace(-np.inf, -100)

    knowledge_df = pd.read_csv('./results/meta-knowledge.csv')
    del knowledge_df['Unnamed: 0']

    knowledge_df = knowledge_df.fillna(0)
    knowledge_df = knowledge_df.replace(np.inf, 100)
    knowledge_df = knowledge_df.replace(-np.inf, -100)

    db = knowledge_df['db']

    # really strong influence over the knn algorithm, maybe it will not be used
    # del knowledge_df['examples']
    # del knowledge_df['attributes']
    # del meta_df['examples']
    # del meta_df['attributes']

    y = knowledge_df[['db','cart_accuracy', 'naive_accuracy', 'neural_accuracy']]
    del knowledge_df['db']
    del knowledge_df['cart_accuracy']
    del knowledge_df['naive_accuracy']
    del knowledge_df['neural_accuracy']
    x = knowledge_df

    d = []

    for i in range(x.count()[0]):
        d2 = 0
        for c in x.columns:
            a1 = meta_df[c].iloc[0]
            a2 = x[c].iloc[i]

            if (c == 'discrete_ratio') and (a1 == 1.0) and (a2 != 1.0):
                d2 = 1000000000000
                break

            d2 = d2 + (a1 - a2) ** 2
        d.append(math.sqrt(d2))

    distances = pd.DataFrame({
        'db': db,
        'distances': d,
        'cart_accuracy': y['cart_accuracy'],
        'naive_accuracy': y['naive_accuracy'],
        'neural_accuracy': y['neural_accuracy']
    })

    distances = distances.sort_values(by=['distances'])

    cart_acc = []
    naive_acc = []
    neural_acc = []

    for i in range(k): # n neighbors
        cart_acc.append(distances['cart_accuracy'].iloc[i])
        naive_acc.append(distances['naive_accuracy'].iloc[i])
        neural_acc.append(distances['neural_accuracy'].iloc[i])

    cart = np.mean(cart_acc)
    naive = np.mean(naive_acc)
    neural = np.mean(neural_acc)

    recomendation = ''

    if cart > naive and cart > neural:
        recomendation = 'cart'
    if naive > cart and naive > neural:
        recomendation = 'naive'
    if neural > cart and neural > naive:
        recomendation = 'neural'

    return recomendation

from sklearn.tree import DecisionTreeClassifier

# the main idea of this script is to run a decision tree for each db

def cart(csv_path, target):

    df = pd.read_csv(csv_path)
    if 'Unnamed: 0' in df.columns:
        del df['Unnamed: 0'] # a small workaround regarding csv exporting issues

    y = df[target]
    del df[target]
    x = df

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

    cart = DecisionTreeClassifier()
    cart = cart.fit(x_train, y_train)
    y_pred = cart.predict(x_test)

    acc = metrics.accuracy_score(y_test, y_pred)

    return acc

from sklearn.naive_bayes import GaussianNB

# the main idea of this script is to run a decision tree for each db

def naive(csv_path, target):

    df = pd.read_csv(csv_path)
    if 'Unnamed: 0' in df.columns:
        del df['Unnamed: 0'] # a small workaround regarding csv exporting issues

    y = df[target]
    del df[target]
    x = df

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

    naive = GaussianNB()
    naive = naive.fit(x_train, y_train)
    y_pred = naive.predict(x_test)

    acc = metrics.accuracy_score(y_test, y_pred)

    return acc

from sklearn.neural_network import MLPClassifier

# the main idea of this script is to run a decision tree for each db

def neural(csv_path, target):

    df = pd.read_csv(csv_path)
    if 'Unnamed: 0' in df.columns:
        del df['Unnamed: 0'] # a small workaround regarding csv exporting issues

    y = df[target]
    del df[target]
    x = df

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

    neural = MLPClassifier(alpha=1e-5,hidden_layer_sizes=(15,), random_state=1)
    neural = neural.fit(x_train, y_train)
    y_pred = neural.predict(x_test)

    acc = metrics.accuracy_score(y_test, y_pred)

    return acc

print('---------------------------------------------------')

# database poker - perfect

poker = pd.read_csv('./test/poker.csv')

print('database poker recomendation: ', knn(poker, 'class'))
print('cart_accuracy', cart('./test/poker.csv', 'class'))
print('naive_accuracy', naive('./test/poker.csv', 'class'))
print('neural_accuracy', neural('./test/poker.csv', 'class'))
print('---------------------------------------------------')

# database agaricus - needs transformation

agaricus = pd.read_csv('./test/agaricus-lepiota.csv')

print('database agaricus recomendation: ', knn(agaricus, 'class'))
print('cart_accuracy', cart('./test/agaricus-lepiota.csv', 'class'))
print('naive_accuracy', naive('./test/agaricus-lepiota.csv', 'class'))
print('neural_accuracy', neural('./test/agaricus-lepiota.csv', 'class'))
print('---------------------------------------------------')

# database car - needs transformation

car = pd.read_csv('./test/car.csv')

print('database car recomendation: ', knn(car, 'class'))
print('cart_accuracy', cart('./test/car.csv', 'class'))
print('naive_accuracy', naive('./test/car.csv', 'class'))
print('neural_accuracy', neural('./test/car.csv', 'class'))
print('---------------------------------------------------')

# database lung - perfect

lung = pd.read_csv('./test/lung-cancer.csv')

print('database lung recomendation: ', knn(lung, 'A1'))
print('cart_accuracy', cart('./test/lung-cancer.csv', 'A1'))
print('naive_accuracy', naive('./test/lung-cancer.csv', 'A1'))
print('neural_accuracy', neural('./test/lung-cancer.csv', 'A1'))
print('---------------------------------------------------')

# database vehicle - needs transformation

vehicle = pd.read_csv('./test/vehicle.csv')

print('database vehicle recomendation: ', knn(vehicle, 'A19'))
print('cart_accuracy', cart('./test/vehicle.csv', 'A19'))
print('naive_accuracy', naive('./test/vehicle.csv', 'A19'))
print('neural_accuracy', neural('./test/vehicle.csv', 'A19'))
print('---------------------------------------------------')

# database messidor - perfect

messidor = pd.read_csv('./test/messidor_features.csv')

print('database messidor recomendation: ', knn(messidor, 'class'))
print('cart_accuracy', cart('./test/messidor_features.csv', 'class'))
print('naive_accuracy', naive('./test/messidor_features.csv', 'class'))
print('neural_accuracy', neural('./test/messidor_features.csv', 'class'))
print('---------------------------------------------------')

# database iris - needs transformation

iris = pd.read_csv('./test/iris.csv')

print('database iris recomendation: ', knn(iris, 'class'))
print('cart_accuracy', cart('./test/iris.csv', 'class'))
print('naive_accuracy', naive('./test/iris.csv', 'class'))
print('neural_accuracy', neural('./test/iris.csv', 'class'))
print('---------------------------------------------------')
