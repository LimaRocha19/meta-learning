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

table = {
    'db': ['abalone', 'adult', 'australian', 'drugs', 'fertility', 'german', 'glass', 'heart', 'ionosphere', 'pendigits', 'phishing', 'failures', 'shuttle', 'spam', 'wdbc', 'wifi', 'wine', 'zoo', 'breast', 'stability', 'student', 'leaf', 'kidney', 'traffic'],
    'cart_accuracy': cart_accuracy,
    'naive_accuracy': naive_accuracy,
    'neural_accuracy': neural_accuracy
}

columns = ['db', 'cart_accuracy', 'naive_accuracy', 'neural_accuracy']

df = pd.DataFrame(table, columns=columns)

df.to_csv('./results/meta-knowledge.csv')
