import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing

def knn(csv_path, target):

    df = pd.read_csv(csv_path)
    del df['Unnamed: 0'] # a small workaround regarding csv exporting issues
    del df['db']

    df = df.fillna(0)
    df = df.replace(np.inf, 100)
    df = df.replace(-np.inf, -100)

    classes = 10
    labels = []
    for i in range(classes):
        labels.append(i)

    y = pd.cut(df[target], classes, labels=labels)
    print(y)
    del df[target]
    x = df

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

    # lab_enc = preprocessing.LabelEncoder()
    # y_train = lab_enc.fit_transform(y_train)
    # y_test = lab_enc.fit_transform(y_test)

    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)

    confusion = metrics.confusion_matrix(y_test, y_pred)
    acc = metrics.accuracy_score(y_test, y_pred)

    return acc

meta_path = './results/meta-knowledge.csv'
cart_target = 'cart_accuracy'
naive_target = 'naive_accuracy'
neural_target = 'neural_accuracy'

print(knn(meta_path, cart_target))
print(knn(meta_path, naive_target))
print(knn(meta_path, neural_target))
