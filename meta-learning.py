import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

def meta_features(csv_path, target):

    df = pd.read_csv(csv_path)
    del df['Unnamed: 0'] # a small workaround regarding csv exporting issues

    # simple characteristics

    # number of classes
    classes = df[target].drop_duplicates().count()

    # number of examples
    examples = df.count()[target]

    # number of features
    features = len(df.columns)

    # number of binary attributes

    # statistical characteristics

    # mean degree of correlation
    pearson = df.corr()[target]
    del pearson[target]
    mean_pearson = pearson.mean()

    # standard deviation
    std = df[target].std()

    # canonical correlation
    canonical = pearson.max()

    # skewness
    skewness = df[target].skew()

    # kurtosis
    kurtosis = df[target].kurtosis()


cart = pd.read_csv('./results/cart.csv')
del cart['Unnamed: 0'] # a small workaround regarding csv exporting issues
cart_accuracy = cart['accuracy']

naive = pd.read_csv('./results/naive.csv')
del naive['Unnamed: 0'] # a small workaround regarding csv exporting issues
naive_accuracy = naive['accuracy']

neural = pd.read_csv('./results/neural.csv')
del neural['Unnamed: 0'] # a small workaround regarding csv exporting issues
neural_accuracy = neural['accuracy']

table = {
    'db': ['abalone', 'adult', 'australian', 'drugs', 'fertility', 'german', 'glass', 'heart', 'ionosphere', 'pendigits', 'phishing', 'failures', 'shuttle', 'spam', 'wdbc', 'wifi', 'wine', 'zoo', 'breast', 'stability', 'student', 'leaf', 'kidney', 'traffic'],
    'cart_accuracy': cart_accuracy,
    'naive_accuracy': naive_accuracy,
    'neural_accuracy': neural_accuracy
}

columns = ['db', 'cart_accuracy', 'naive_accuracy', 'neural_accuracy']

df = pd.DataFrame(table, columns=columns)

df.to_csv('./results/meta-knowledge.csv')
