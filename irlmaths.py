import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from pandas import ExcelWriter
from pandas import ExcelFile

from scipy import stats

# works for pandas dataframe

def outliers(df):

    outliers = []

    threshold = 3
    mean = df.mean()
    std = df.std()

    for i in df:
        z = (i - mean) / std
        if np.abs(z) > threshold:
            outliers.append(i)

    return outliers

def metadata(df, path):

    df_examples = df.count()[0]
    df_attributes = len(df.columns)
    df_discrete_ratio = len(df.select_dtypes(include=['int64']).columns) / df_attributes

    entropies = []

    for column in df.select_dtypes(include=['int64']).columns:
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
        'mean_kurtosis': df_mean_kurtosis
    }

    columns = ['examples', 'attributes', 'discrete_ratio', 'mean_entropy', 'mean_correlation', 'mean_skew', 'mean_kurtosis']

    meta_df = pd.DataFrame(table, columns=columns, index=[0])

    meta_df.to_csv(path)
