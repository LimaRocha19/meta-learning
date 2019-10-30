import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plotter(csv_path, target):

    df = pd.read_csv(csv_path)
    del df['Unnamed: 0']
    del df['db']

    for column in df.columns:

        table = pd.DataFrame({column: df[column], target: df[target]})
        table = table.replace([np.inf, -np.inf], np.nan)
        table = table.dropna()

        for index, row in table.iterrows():
            z = (row[column] - table[column].mean()) / table[column].std()
            if np.abs(z) > 3:
                table = table.drop(index)

        plt.scatter(table[column], table[target])
        plt.title(column + ' vs ' + target)
        plt.xlabel(column)
        plt.ylabel(target)
        plt.savefig('./plots/' + column + '-' + target + '.png')
        plt.clf()

# database abalone

meta_path = './results/meta-knowledge.csv'
cart_target = 'cart_accuracy'
naive_target = 'naive_accuracy'
neural_target = 'neural_accuracy'

plotter(meta_path, cart_target)
plotter(meta_path, naive_target)
plotter(meta_path, neural_target)
