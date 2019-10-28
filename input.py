import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from pandas import ExcelWriter
from pandas import ExcelFile

from scipy import stats

from irlmaths import metadata

# the main idea of this script is to replace missing values and transform every feature to numeric

# database abalone - needs to transform values

abalone = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='ABALONE')

abalone['Sex'] = pd.factorize(abalone['Sex'])[0]

abalone.to_csv('./csv/abalone.csv')
metadata(abalone, './metadata/abalone.csv', 'Sex')

# database adult - needs to replace missing value and transform values

adult = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='ADULT')

adult['workclass'] = adult['workclass'].replace('?', adult['workclass'].value_counts().idxmax())
adult['occupation'] = adult['occupation'].replace('?', adult['occupation'].value_counts().idxmax())
adult['native-country'] = adult['native-country'].replace('?', adult['native-country'].value_counts().idxmax())

adult['workclass'] = pd.factorize(adult['workclass'])[0]
adult['education'] = pd.factorize(adult['education'])[0]
adult['marital-status'] = pd.factorize(adult['marital-status'])[0]
adult['occupation'] = pd.factorize(adult['occupation'])[0]
adult['relationship'] = pd.factorize(adult['relationship'])[0]
adult['race'] = pd.factorize(adult['race'])[0]
adult['sex'] = pd.factorize(adult['sex'])[0]
adult['native-country'] = pd.factorize(adult['native-country'])[0]
adult['workclass'] = pd.factorize(adult['workclass'])[0]
adult['income'] = pd.factorize(adult['income'])[0]

adult.to_csv('./csv/adult.csv')
metadata(adult, './metadata/adult.csv', 'income')

# database australian - needs to transform values

australian = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='AUSTRALIAN')

australian.to_csv('./csv/australian.csv')
metadata(australian, './metadata/australian.csv', 'A15')

# database drugs - needs to transform values

drugs = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='DRUGS')

del drugs['ID']

drugs['Age'] = pd.factorize(drugs['Age'])[0]
drugs['Gender'] = pd.factorize(drugs['Gender'])[0]
drugs['Education'] = pd.factorize(drugs['Education'])[0]
drugs['Country'] = pd.factorize(drugs['Country'])[0]
drugs['Ethnicity'] = pd.factorize(drugs['Ethnicity'])[0]
drugs['Nscore'] = pd.factorize(drugs['Nscore'])[0]
drugs['Escore'] = pd.factorize(drugs['Escore'])[0]
drugs['Oscore'] = pd.factorize(drugs['Oscore'])[0]
drugs['Ascore'] = pd.factorize(drugs['Ascore'])[0]
drugs['Cscore'] = pd.factorize(drugs['Cscore'])[0]
drugs['Impulsive'] = pd.factorize(drugs['Impulsive'])[0]
drugs['SS'] = pd.factorize(drugs['SS'])[0]

drugs['Alcohol'] = pd.factorize(drugs['Alcohol'])[0]
drugs['Amphet'] = pd.factorize(drugs['Amphet'])[0]
drugs['Amyl'] = pd.factorize(drugs['Amyl'])[0]
drugs['Benzos'] = pd.factorize(drugs['Benzos'])[0]
drugs['Caff'] = pd.factorize(drugs['Caff'])[0]
drugs['Cannabis'] = pd.factorize(drugs['Cannabis'])[0]
drugs['Choc'] = pd.factorize(drugs['Choc'])[0]
drugs['Coke'] = pd.factorize(drugs['Coke'])[0]
drugs['Crack'] = pd.factorize(drugs['Crack'])[0]
drugs['Ecstasy'] = pd.factorize(drugs['Ecstasy'])[0]
drugs['Heroin'] = pd.factorize(drugs['Heroin'])[0]
drugs['Ketamine'] = pd.factorize(drugs['Ketamine'])[0]
drugs['Legalh'] = pd.factorize(drugs['Legalh'])[0]
drugs['LSD'] = pd.factorize(drugs['LSD'])[0]
drugs['Meth'] = pd.factorize(drugs['Meth'])[0]
drugs['Mushrooms'] = pd.factorize(drugs['Mushrooms'])[0]
drugs['Nicotine'] = pd.factorize(drugs['Nicotine'])[0]
drugs['Semer'] = pd.factorize(drugs['Semer'])[0]
drugs['VSA'] = pd.factorize(drugs['VSA'])[0]

drugs.to_csv('./csv/drugs.csv')
metadata(drugs, './metadata/drugs.csv', 'Alcohol')

# database fertility - needs to transform values

fertility = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='FERTILITY')

fertility['Season'] = pd.factorize(fertility['Season'])[0]
fertility['Age'] = pd.factorize(fertility['Age'])[0]
fertility['Childish Diseases'] = pd.factorize(fertility['Childish Diseases'])[0]
fertility['Accident or Trauma'] = pd.factorize(fertility['Accident or Trauma'])[0]
fertility['Surgical Intervention'] = pd.factorize(fertility['Surgical Intervention'])[0]
fertility['High Fevers Last Year'] = pd.factorize(fertility['High Fevers Last Year'])[0]
fertility['Alcohol Frequency'] = pd.factorize(fertility['Alcohol Frequency'])[0]
fertility['Smoking'] = pd.factorize(fertility['Smoking'])[0]
fertility['Output'] = pd.factorize(fertility['Output'])[0]

fertility.to_csv('./csv/fertility.csv')
metadata(fertility, './metadata/fertility.csv', 'Output')

# database german - needs nothing (maybe little transformation)

german = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='GERMAN')

german.to_csv('./csv/german.csv')
metadata(german, './metadata/german.csv', 'A25')

# database glass - needs nothing (maybe little transformation)

glass = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='GLASS')

del glass['ID']

glass.to_csv('./csv/glass.csv')
metadata(glass, './metadata/glass.csv', 'Type')

# database heart - needs nothing (maybe little transformation)

heart = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='HEART')

heart.to_csv('./csv/heart.csv')
metadata(heart, './metadata/heart.csv', 'A14')

# database ionosphere - needs to transform values

ionosphere = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='IONOSPHERE')

del ionosphere['A2']

ionosphere['A35'] = pd.factorize(ionosphere['A35'])[0]

ionosphere.to_csv('./csv/ionosphere.csv')
metadata(ionosphere, './metadata/ionosphere.csv', 'A35')

# database pendigits - needs nothing (maybe little transformation)

pendigits = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='PENDIGITS')

pendigits.to_csv('./csv/pendigits.csv')
metadata(pendigits, './metadata/pendigits.csv', 'A17')

# database phishing - needs nothing (maybe little transformation)

phishing = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='PHISHING')

for column in phishing.columns:
    phishing[column] = pd.factorize(phishing[column])[0]

phishing.to_csv('./csv/phishing.csv')
metadata(phishing, './metadata/phishing.csv', 'A31')

# database failures - needs nothing (maybe little transformation)

failures = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='FAILURES')

del failures['Run']

failures.to_csv('./csv/failures.csv')
metadata(failures, './metadata/failures.csv', 'outcome')

# database shuttle - needs nothing (maybe little transformation)

shuttle = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='SHUTTLE')

for column in shuttle.columns:
    shuttle[column] = pd.factorize(shuttle[column])[0]

shuttle.to_csv('./csv/shuttle.csv')
metadata(shuttle, './metadata/shuttle.csv', 'A10')

# database spam - needs nothing (maybe little transformation)

spam = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='SPAM')

spam.to_csv('./csv/spam.csv')
metadata(spam, './metadata/spam.csv', 'A58')

# database wdbc - needs to transform values

wdbc = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='WDBC')

del wdbc['ID']

wdbc['Cancer'] = pd.factorize(wdbc['Cancer'])[0]

wdbc.to_csv('./csv/wdbc.csv')
metadata(wdbc, './metadata/wdbc.csv', 'Cancer')

# database wifi - needs nothing (maybe little transformation)

wifi = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='WIFI')

wifi.to_csv('./csv/wifi.csv')
metadata(wifi, './metadata/wifi.csv', 'A8')

# database wine - needs nothing (maybe little transformation)

wine = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='WINE')

wine.to_csv('./csv/wine.csv')
metadata(wine, './metadata/wine.csv', 'Class')

# database zoo - needs to transform values

zoo = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='ZOO')

zoo['Name'] = pd.factorize(zoo['Name'])[0]

zoo.to_csv('./csv/zoo.csv')
metadata(zoo, './metadata/zoo.csv', 'type')

# database breast - needs to transform values

breast = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='BREAST')

del breast['Case #']

breast['Class'] = pd.factorize(breast['Class'])[0]

breast.to_csv('./csv/breast.csv')
metadata(breast, './metadata/breast.csv', 'Class')

# database stability - needs to transform values

stability = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='STABILITY')

stability['stabf'] = pd.factorize(stability['stabf'])[0]

stability.to_csv('./csv/stability.csv')
metadata(stability, './metadata/stability.csv', 'stabf')

# database trip - needs nothing (maybe little transformation)

# trip = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='TRIP')
#
# del trip['User ID']
#
# trip.to_csv('./csv/trip.csv')
# metadata(trip, './metadata/trip.csv')

# database student - needs to transform values

student = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='STUDENT')

student['school'] = pd.factorize(student['school'])[0]
student['sex'] = pd.factorize(student['sex'])[0]
student['address'] = pd.factorize(student['address'])[0]
student['famsize'] = pd.factorize(student['famsize'])[0]
student['pstatus'] = pd.factorize(student['pstatus'])[0]
student['mjob'] = pd.factorize(student['mjob'])[0]
student['fjob'] = pd.factorize(student['fjob'])[0]
student['reason'] = pd.factorize(student['reason'])[0]
student['guardian'] = pd.factorize(student['guardian'])[0]
student['schoolsup'] = pd.factorize(student['schoolsup'])[0]
student['famsup'] = pd.factorize(student['famsup'])[0]
student['paid'] = pd.factorize(student['paid'])[0]
student['activities'] = pd.factorize(student['activities'])[0]
student['nursery'] = pd.factorize(student['nursery'])[0]
student['higher'] = pd.factorize(student['higher'])[0]
student['internet'] = pd.factorize(student['internet'])[0]
student['romantic'] = pd.factorize(student['romantic'])[0]

student.to_csv('./csv/student.csv')
metadata(student, './metadata/student.csv', 'approve')

# database leaf - needs nothing (maybe little transformation)

leaf = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='LEAF')

leaf.to_csv('./csv/leaf.csv')
metadata(leaf, './metadata/leaf.csv', 'Class')

# database kidney - needs to replace missing value and transform values

kidney = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='KIDNEY')

kidney['age'] = kidney['age'].replace('?', round(kidney[kidney['age'] != '?']['age'].mean()))
kidney['bp'] = kidney['bp'].replace('?', round(kidney[kidney['bp'] != '?']['bp'].mean()))
kidney['sg'] = kidney['sg'].replace('?', kidney[kidney['sg'] != '?']['sg'].value_counts().idxmax())
kidney['al'] = kidney['al'].replace('?', kidney[kidney['al'] != '?']['al'].value_counts().idxmax())
kidney['su'] = kidney['su'].replace('?', kidney[kidney['su'] != '?']['su'].value_counts().idxmax())
kidney['rbc'] = kidney['rbc'].replace('?', kidney[kidney['rbc'] != '?']['rbc'].value_counts().idxmax())
kidney['pc'] = kidney['pc'].replace('?', kidney[kidney['pc'] != '?']['pc'].value_counts().idxmax())
kidney['pcc'] = kidney['pcc'].replace('?', kidney[kidney['pcc'] != '?']['pcc'].value_counts().idxmax())
kidney['ba'] = kidney['ba'].replace('?', kidney[kidney['ba'] != '?']['ba'].value_counts().idxmax())
kidney['bgr'] = kidney['bgr'].replace('?', round(kidney[kidney['bgr'] != '?']['bgr'].mean()))
kidney['bu'] = kidney['bu'].replace('?', round(kidney[kidney['bu'] != '?']['bu'].mean()))
kidney['sc'] = kidney['sc'].replace('?', round(kidney[kidney['sc'] != '?']['sc'].mean(), 1))
kidney['sod'] = kidney['sod'].replace('?', round(kidney[kidney['sod'] != '?']['sod'].mean()))
kidney['pot'] = kidney['pot'].replace('?', round(kidney[kidney['pot'] != '?']['pot'].mean(), 1))
kidney['hemo'] = kidney['hemo'].replace('?', round(kidney[kidney['hemo'] != '?']['hemo'].mean(), 1))
kidney['pcv'] = kidney['pcv'].replace('?', round(kidney[kidney['pcv'] != '?']['pcv'].mean()))
kidney['wbcc'] = kidney['wbcc'].replace('?', round(kidney[kidney['wbcc'] != '?']['wbcc'].mean()))
kidney['rbcc'] = kidney['rbcc'].replace('?', round(kidney[kidney['rbcc'] != '?']['rbcc'].mean(), 1))
kidney['htn'] = kidney['htn'].replace('?', kidney[kidney['htn'] != '?']['htn'].value_counts().idxmax())
kidney['dm'] = kidney['dm'].replace('?', kidney[kidney['dm'] != '?']['dm'].value_counts().idxmax())
kidney['cad'] = kidney['cad'].replace('?', kidney[kidney['cad'] != '?']['cad'].value_counts().idxmax())
kidney['appet'] = kidney['appet'].replace('?', kidney[kidney['appet'] != '?']['appet'].value_counts().idxmax())
kidney['pe'] = kidney['pe'].replace('?', kidney[kidney['pe'] != '?']['pe'].value_counts().idxmax())
kidney['ane'] = kidney['ane'].replace('?', kidney[kidney['ane'] != '?']['ane'].value_counts().idxmax())

kidney['rbc'] = pd.factorize(kidney['rbc'])[0]
kidney['pc'] = pd.factorize(kidney['pc'])[0]
kidney['pcc'] = pd.factorize(kidney['pcc'])[0]
kidney['ba'] = pd.factorize(kidney['ba'])[0]
kidney['htn'] = pd.factorize(kidney['htn'])[0]
kidney['dm'] = pd.factorize(kidney['dm'])[0]
kidney['cad'] = pd.factorize(kidney['cad'])[0]
kidney['appet'] = pd.factorize(kidney['appet'])[0]
kidney['pe'] = pd.factorize(kidney['pe'])[0]
kidney['ane'] = pd.factorize(kidney['ane'])[0]
kidney['class'] = pd.factorize(kidney['class'])[0]

kidney.to_csv('./csv/kidney.csv')
metadata(kidney, './metadata/kidney.csv', 'class')

# database traffic - needs nothing (maybe little transformation)

traffic = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='TRAFFIC')

traffic['Slowness in traffic (%)'] = pd.factorize(traffic['Slowness in traffic (%)'])[0]

traffic.to_csv('./csv/traffic.csv')
metadata(traffic, './metadata/traffic.csv', 'Slowness in traffic (%)')
