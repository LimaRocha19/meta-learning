import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


cart = pd.read_csv('./results/cart.csv')
del cart['Unnamed: 0'] # a small workaround regarding csv exporting issues
cart_accuracy = cart['accuracy']

naive = pd.read_csv('./results/naive.csv')
del naive['Unnamed: 0'] # a small workaround regarding csv exporting issues
naive_accuracy = naive['accuracy']

neural = pd.read_csv('./results/neural.csv')
del neural['Unnamed: 0'] # a small workaround regarding csv exporting issues
neural_accuracy = neural['accuracy']

db = ['abalone', 'adult', 'australian', 'drugs', 'fertility', 'german', 'glass', 'heart', 'ionosphere', 'pendigits', 'phishing', 'failures', 'shuttle', 'spam', 'wdbc', 'wifi', 'wine', 'zoo', 'breast', 'stability', 'student', 'leaf', 'kidney', 'traffic']

examples = []
attributes = []
discrete_ratio = []
mean_entropy = []
mean_correlation = []
mean_skew = []
mean_kurtosis = []
outliers = []
classes = []
entropy = []

for d in db:
    meta = pd.read_csv('./metadata/' + d + '.csv')
    del meta['Unnamed: 0']
    examples.append(meta['examples'][0])
    attributes.append(meta['attributes'][0])
    discrete_ratio.append(meta['discrete_ratio'][0])
    mean_entropy.append(meta['mean_entropy'][0])
    mean_correlation.append(meta['mean_correlation'][0])
    mean_skew.append(meta['mean_skew'][0])
    mean_kurtosis.append(meta['mean_kurtosis'][0])
    outliers.append(meta['outliers'][0])
    classes.append(meta['classes'][0])
    entropy.append(meta['entropy'][0])

table = {
    'db': db,
    'examples': examples,
    'attributes': attributes,
    'discrete_ratio': discrete_ratio,
    'mean_entropy': mean_entropy,
    'mean_correlation': mean_correlation,
    'mean_skew': mean_skew,
    'mean_kurtosis': mean_kurtosis,
    'outliers': outliers,
    'classes': classes,
    'entropy': entropy,
    'cart_accuracy': cart_accuracy,
    'naive_accuracy': naive_accuracy,
    'neural_accuracy': neural_accuracy
}

columns = ['db', 'examples', 'attributes', 'discrete_ratio', 'mean_entropy', 'mean_correlation', 'mean_skew', 'mean_kurtosis', 'outliers', 'classes', 'entropy', 'cart_accuracy', 'naive_accuracy', 'neural_accuracy']

df = pd.DataFrame(table, columns=columns)

df.to_csv('./results/meta-knowledge.csv')
