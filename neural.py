import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# the main idea of this script is to run a decision tree for each db

def neural(csv_path, target):

    df = pd.read_csv(csv_path)
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

# i could make a loop to automate this, but i have to specify each target

# database abalone

abalone_path = './csv/abalone.csv'
abalone_target = 'Sex'
abalone_acc = neural(abalone_path, abalone_target)

# database adult

adult_path = './csv/adult.csv'
adult_target = 'income'
adult_acc = neural(adult_path, adult_target)

# database australian

australian_path = './csv/australian.csv'
australian_target = 'A15'
australian_acc = neural(australian_path, australian_target)

# database drugs

drugs_path = './csv/drugs.csv'
drugs_target = 'Alcohol'
drugs_acc = neural(drugs_path, drugs_target)

# database fertility

fertility_path = './csv/fertility.csv'
fertility_target = 'Output'
fertility_acc = neural(fertility_path, fertility_target)

# database german

german_path = './csv/german.csv'
german_target = 'A25'
german_acc = neural(german_path, german_target)

# database glass

glass_path = './csv/glass.csv'
glass_target = 'Type'
glass_acc = neural(glass_path, glass_target)

# database heart

heart_path = './csv/heart.csv'
heart_target = 'A14'
heart_acc = neural(heart_path, heart_target)

# database ionosphere

ionosphere_path = './csv/ionosphere.csv'
ionosphere_target = 'A35'
ionosphere_acc = neural(ionosphere_path, ionosphere_target)

# database pendigits

pendigits_path = './csv/pendigits.csv'
pendigits_target = 'A17'
pendigits_acc = neural(pendigits_path, pendigits_target)

# database phishing

phishing_path = './csv/phishing.csv'
phishing_target = 'A31'
phishing_acc = neural(phishing_path, phishing_target)

# database failures

failures_path = './csv/failures.csv'
failures_target = 'outcome'
failures_acc = neural(failures_path, failures_target)

# database shuttle

shuttle_path = './csv/shuttle.csv'
shuttle_target = 'A10'
shuttle_acc = neural(shuttle_path, shuttle_target)

# database spam

spam_path = './csv/spam.csv'
spam_target = 'A58'
spam_acc = neural(spam_path, spam_target)

# database wdbc

wdbc_path = './csv/wdbc.csv'
wdbc_target = 'Cancer'
wdbc_acc = neural(wdbc_path, wdbc_target)

# database wifi

wifi_path = './csv/wifi.csv'
wifi_target = 'A8'
wifi_acc = neural(wifi_path, wifi_target)

# database wine

wine_path = './csv/wine.csv'
wine_target = 'Class'
wine_acc = neural(wine_path, wine_target)

# database zoo

zoo_path = './csv/zoo.csv'
zoo_target = 'type'
zoo_acc = neural(zoo_path, zoo_target)

# database breast

breast_path = './csv/breast.csv'
breast_target = 'Class'
breast_acc = neural(breast_path, breast_target)

# database stability

stability_path = './csv/stability.csv'
stability_target = 'stabf'
stability_acc = neural(stability_path, stability_target)

# database trip - WILL BE ANALYZED BETTER SOON

# database student

student_path = './csv/student.csv'
student_target = 'approve'
student_acc = neural(student_path, student_target)

# database leaf

leaf_path = './csv/leaf.csv'
leaf_target = 'Class'
leaf_acc = neural(leaf_path, leaf_target)

# database kidney

kidney_path = './csv/kidney.csv'
kidney_target = 'class'
kidney_acc = neural(kidney_path, kidney_target)

# database traffic

traffic_path = './csv/traffic.csv'
traffic_target = 'Slowness in traffic (%)'
traffic_acc = neural(traffic_path, traffic_target)

# creating a dataframe containing the accuracy from each db processed with neural

table = {
    'db': [abalone_path, adult_path, australian_path, drugs_path, fertility_path, german_path, glass_path, heart_path, ionosphere_path, pendigits_path, phishing_path, failures_path, shuttle_path, spam_path, wdbc_path, wifi_path, wine_path, zoo_path, breast_path, stability_path, student_path, leaf_path, kidney_path, traffic_path],
    'accuracy': [abalone_acc, adult_acc, australian_acc, drugs_acc, fertility_acc, german_acc, glass_acc, heart_acc, ionosphere_acc, pendigits_acc, phishing_acc, failures_acc, shuttle_acc, spam_acc, wdbc_acc, wifi_acc, wine_acc, zoo_acc, breast_acc, stability_acc, student_acc, leaf_acc, kidney_acc, traffic_acc]
}

columns = ['db', 'accuracy']

df = pd.DataFrame(table, columns=columns)

df.to_csv('./results/neural.csv')
