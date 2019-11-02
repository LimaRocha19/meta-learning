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

def knn(df, target):

    if 'Unnamed: 0' in df.columns:
        del df['Unnamed: 0'] # a small workaround regarding csv exporting issues

    # metadata extraction for the new database

    # print(df.count())

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

    # print(meta_df)

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

    return distances

def recomend(distances, k=3):

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

# database poker - perfect

poker = pd.read_csv('./test/poker.csv')

poker_cart = cart('./test/poker.csv', 'class')
poker_naive = naive('./test/poker.csv', 'class')
poker_neural = neural('./test/poker.csv', 'class')

poker_reccomended = 'cart' if (poker_cart > poker_naive and poker_cart > poker_neural) else ('naive' if (poker_naive > poker_cart and poker_naive > poker_neural) else 'neural')

# database agaricus - needs transformation

agaricus = pd.read_csv('./test/agaricus-lepiota.csv')

agaricus_cart = cart('./test/agaricus-lepiota.csv', 'class')
agaricus_naive = naive('./test/agaricus-lepiota.csv', 'class')
agaricus_neural = neural('./test/agaricus-lepiota.csv', 'class')

agaricus_reccomended = 'cart' if (agaricus_cart > agaricus_naive and agaricus_cart > agaricus_neural) else ('naive' if (agaricus_naive > agaricus_cart and agaricus_naive > agaricus_neural) else 'neural')

# database car - needs transformation

car = pd.read_csv('./test/car.csv')

car_cart = cart('./test/car.csv', 'class')
car_naive = naive('./test/car.csv', 'class')
car_neural = neural('./test/car.csv', 'class')

car_reccomended = 'cart' if (car_cart > car_naive and car_cart > car_neural) else ('naive' if (car_naive > car_cart and car_naive > car_neural) else 'neural')

# database lung - perfect

lung = pd.read_csv('./test/lung-cancer.csv')

lung_cart = cart('./test/lung-cancer.csv', 'A1')
lung_naive = naive('./test/lung-cancer.csv', 'A1')
lung_neural = neural('./test/lung-cancer.csv', 'A1')

lung_reccomended = 'cart' if (lung_cart > lung_naive and lung_cart > lung_neural) else ('naive' if (lung_naive > lung_cart and lung_naive > lung_neural) else 'neural')

# database vehicle - needs transformation

vehicle = pd.read_csv('./test/vehicle.csv')

vehicle_cart = cart('./test/vehicle.csv', 'A19')
vehicle_naive = naive('./test/vehicle.csv', 'A19')
vehicle_neural = neural('./test/vehicle.csv', 'A19')

vehicle_reccomended = 'cart' if (vehicle_cart > vehicle_naive and vehicle_cart > vehicle_neural) else ('naive' if (vehicle_naive > vehicle_cart and vehicle_naive > vehicle_neural) else 'neural')

# database messidor - perfect

messidor = pd.read_csv('./test/messidor_features.csv')

messidor_cart = cart('./test/messidor_features.csv', 'class')
messidor_naive = naive('./test/messidor_features.csv', 'class')
messidor_neural = neural('./test/messidor_features.csv', 'class')

messidor_reccomended = 'cart' if (messidor_cart > messidor_naive and messidor_cart > messidor_neural) else ('naive' if (messidor_naive > messidor_cart and messidor_naive > messidor_neural) else 'neural')

# database iris - needs transformation

iris = pd.read_csv('./test/iris.csv')

iris_cart = cart('./test/iris.csv', 'class')
iris_naive = naive('./test/iris.csv', 'class')
iris_neural = neural('./test/iris.csv', 'class')

iris_reccomended = 'cart' if (iris_cart > iris_naive and iris_cart > iris_neural) else ('naive' if (iris_naive > iris_cart and iris_naive > iris_neural) else 'neural')

# predicting best algorithm

real = [poker_reccomended, agaricus_reccomended, car_reccomended, lung_reccomended, vehicle_reccomended, messidor_reccomended, iris_reccomended]

neighbors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
accuracies = []

poker_d = knn(poker, 'class')
agaricus_d = knn(agaricus, 'class')
car_d = knn(car, 'class')
lung_d = knn(lung, 'A1')
vehicle_d = knn(vehicle, 'A19')
messidor_d = knn(messidor, 'class')
iris_d = knn(iris, 'class')

for k in neighbors:

    print('!!!!!!!!!!!!!!!!!!!!!!!!! Neighbors:  ')
    print(k)

    poker_predicted = recomend(poker_d, k)
    agaricus_predicted = recomend(agaricus_d, k)
    car_predicted = recomend(car_d, k)
    lung_predicted = recomend(lung_d, k)
    vehicle_predicted = recomend(vehicle_d, k)
    messidor_predicted = recomend(messidor_d, k)
    iris_predicted = recomend(iris_d, k)

    pred = [poker_predicted, agaricus_predicted, car_predicted, lung_predicted, vehicle_predicted, messidor_predicted, iris_predicted]

    acc = metrics.accuracy_score(real, pred)
    accuracies.append(acc)

plt.scatter(neighbors, accuracies)
plt.title('neighbors vs accuracy')
plt.xlabel('neighbors')
plt.ylabel('accuracy')
plt.plot(neighbors, accuracies, '-o')
plt.grid()
plt.show()
