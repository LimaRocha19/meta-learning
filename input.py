import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas import ExcelWriter
from pandas import ExcelFile

# database abalone - needs to transform values

# abalone = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='ABALONE')
#
# abalone['Sex'] = pd.factorize(abalone['Sex'])[0]
#
# abalone.to_csv('abalone.csv')

# database adult - needs to replace missing value and transform values

# adult = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='ADULT')
#
# adult['workclass'] = adult['workclass'].replace('?', adult['workclass'].value_counts().idxmax())
# adult['occupation'] = adult['occupation'].replace('?', adult['occupation'].value_counts().idxmax())
# adult['native-country'] = adult['native-country'].replace('?', adult['native-country'].value_counts().idxmax())
#
# adult['workclass'] = pd.factorize(adult['workclass'])[0]
# adult['education'] = pd.factorize(adult['education'])[0]
# adult['marital-status'] = pd.factorize(adult['marital-status'])[0]
# adult['occupation'] = pd.factorize(adult['occupation'])[0]
# adult['relationship'] = pd.factorize(adult['relationship'])[0]
# adult['race'] = pd.factorize(adult['race'])[0]
# adult['sex'] = pd.factorize(adult['sex'])[0]
# adult['native-country'] = pd.factorize(adult['native-country'])[0]
# adult['workclass'] = pd.factorize(adult['workclass'])[0]
# adult['income'] = pd.factorize(adult['income'])[0]
#
# adult.to_csv('adult.csv')

# database australian - needs to transform values

# australian = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='AUSTRALIAN')
#
# australian.to_csv('australian.csv')

# database drugs - needs to transform values

# drugs = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='DRUGS')
#
# drugs['Alcohol'] = pd.factorize(drugs['Alcohol'])[0]
# drugs['Amphet'] = pd.factorize(drugs['Amphet'])[0]
# drugs['Amyl'] = pd.factorize(drugs['Amyl'])[0]
# drugs['Benzos'] = pd.factorize(drugs['Benzos'])[0]
# drugs['Caff'] = pd.factorize(drugs['Caff'])[0]
# drugs['Cannabis'] = pd.factorize(drugs['Cannabis'])[0]
# drugs['Choc'] = pd.factorize(drugs['Choc'])[0]
# drugs['Coke'] = pd.factorize(drugs['Coke'])[0]
# drugs['Crack'] = pd.factorize(drugs['Crack'])[0]
# drugs['Ecstasy'] = pd.factorize(drugs['Ecstasy'])[0]
# drugs['Heroin'] = pd.factorize(drugs['Heroin'])[0]
# drugs['Ketamine'] = pd.factorize(drugs['Ketamine'])[0]
# drugs['Legalh'] = pd.factorize(drugs['Legalh'])[0]
# drugs['LSD'] = pd.factorize(drugs['LSD'])[0]
# drugs['Meth'] = pd.factorize(drugs['Meth'])[0]
# drugs['Mushrooms'] = pd.factorize(drugs['Mushrooms'])[0]
# drugs['Nicotine'] = pd.factorize(drugs['Nicotine'])[0]
# drugs['Semer'] = pd.factorize(drugs['Semer'])[0]
# drugs['VSA'] = pd.factorize(drugs['VSA'])[0]
#
# drugs.to_csv('drugs.csv')

# database fertility - needs to transform values

# fertility = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='FERTILITY')
#
# fertility['Output'] = pd.factorize(fertility['Output'])[0]
#
# fertility.to_csv('fertility.csv')

# database german - needs nothing (maybe little transformation)

# german = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='GERMAN')
#
# german.to_csv('german.csv')

# database glass - needs nothing (maybe little transformation)

# glass = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='GLASS')
#
# glass.to_csv('glass.csv')

# database heart - needs nothing (maybe little transformation)

# heart = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='HEART')
#
# heart.to_csv('heart.csv')

# database ionosphere - needs to transform values

# ionosphere = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='IONOSPHERE')
#
# ionosphere['A35'] = pd.factorize(ionosphere['A35'])[0]
#
# ionosphere.to_csv('ionosphere.csv')

# database pendigits - needs nothing (maybe little transformation)

# pendigits = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='PENDIGITS')
#
# pendigits.to_csv('pendigits.csv')

# database phishing - needs nothing (maybe little transformation)

# phishing = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='PHISHING')
#
# phishing.to_csv('phishing.csv')

# database failures - needs nothing (maybe little transformation)

# failures = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='FAILURES')
#
# failures.to_csv('failures.csv')

# database shuttle - needs nothing (maybe little transformation)

# shuttle = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='SHUTTLE')
#
# shuttle.to_csv('shuttle.csv')

# database spam - needs nothing (maybe little transformation)

# spam = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='SPAM')
#
# spam.to_csv('spam.csv')

# database wdbc - needs to transform values

# wdbc = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='WDBC')
#
# wdbc['Cancer'] = pd.factorize(wdbc['Cancer'])[0]
#
# wdbc.to_csv('wdbc.csv')

# database wifi - needs nothing (maybe little transformation)

# wifi = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='WIFI')
#
# wifi.to_csv('wifi.csv')

# database wine - needs nothing (maybe little transformation)

# wine = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='WINE')
#
# wine.to_csv('wine.csv')

# database zoo - needs to transform values

# zoo = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='ZOO')
#
# zoo['Name'] = pd.factorize(zoo['Name'])[0]
#
# zoo.to_csv('zoo.csv')

# database breast - needs to transform values

# breast = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='BREAST')
#
# breast['Class'] = pd.factorize(breast['Class'])[0]
#
# breast.to_csv('breast.csv')

# database stability - needs to transform values

# stability = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='STABILITY')
#
# stability['stabf'] = pd.factorize(stability['stabf'])[0]
#
# stability.to_csv('stability.csv')

# database trip - needs nothing (maybe little transformation)

# trip = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='TRIP')
#
# trip.to_csv('trip.csv')

# database student - needs to transform values

# student = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='STUDENT')
#
# student['school'] = pd.factorize(student['school'])[0]
# student['sex'] = pd.factorize(student['sex'])[0]
# student['address'] = pd.factorize(student['address'])[0]
# student['famsize'] = pd.factorize(student['famsize'])[0]
# student['pstatus'] = pd.factorize(student['pstatus'])[0]
# student['mjob'] = pd.factorize(student['mjob'])[0]
# student['fjob'] = pd.factorize(student['fjob'])[0]
# student['reason'] = pd.factorize(student['reason'])[0]
# student['guardian'] = pd.factorize(student['guardian'])[0]
# student['schoolsup'] = pd.factorize(student['schoolsup'])[0]
# student['famsup'] = pd.factorize(student['famsup'])[0]
# student['paid'] = pd.factorize(student['paid'])[0]
# student['activities'] = pd.factorize(student['activities'])[0]
# student['nursery'] = pd.factorize(student['nursery'])[0]
# student['higher'] = pd.factorize(student['higher'])[0]
# student['internet'] = pd.factorize(student['internet'])[0]
# student['romantic'] = pd.factorize(student['romantic'])[0]
#
# student.to_csv('student.csv')

# database leaf - needs nothing (maybe little transformation)

# leaf = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='LEAF')
#
# leaf.to_csv('leaf.csv')

# database kidney - needs to replace missing value and transform values

# kidney = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='KIDNEY')
#
# kidney['age'] = kidney['age'].replace('?', round(kidney[kidney['age'] != '?']['age'].mean()))
# kidney['bp'] = kidney['bp'].replace('?', round(kidney[kidney['bp'] != '?']['bp'].mean()))
# kidney['sg'] = kidney['sg'].replace('?', kidney[kidney['sg'] != '?']['sg'].value_counts().idxmax())
# kidney['al'] = kidney['al'].replace('?', kidney[kidney['al'] != '?']['al'].value_counts().idxmax())
# kidney['su'] = kidney['su'].replace('?', kidney[kidney['su'] != '?']['su'].value_counts().idxmax())
# kidney['rbc'] = kidney['rbc'].replace('?', kidney[kidney['rbc'] != '?']['rbc'].value_counts().idxmax())
# kidney['pc'] = kidney['pc'].replace('?', kidney[kidney['pc'] != '?']['pc'].value_counts().idxmax())
# kidney['pcc'] = kidney['pcc'].replace('?', kidney[kidney['pcc'] != '?']['pcc'].value_counts().idxmax())
# kidney['ba'] = kidney['ba'].replace('?', kidney[kidney['ba'] != '?']['ba'].value_counts().idxmax())
# kidney['bgr'] = kidney['bgr'].replace('?', round(kidney[kidney['bgr'] != '?']['bgr'].mean()))
# kidney['bu'] = kidney['bu'].replace('?', round(kidney[kidney['bu'] != '?']['bu'].mean()))
# kidney['sc'] = kidney['sc'].replace('?', round(kidney[kidney['sc'] != '?']['sc'].mean(), 1))
# kidney['sod'] = kidney['sod'].replace('?', round(kidney[kidney['sod'] != '?']['sod'].mean()))
# kidney['pot'] = kidney['pot'].replace('?', round(kidney[kidney['pot'] != '?']['pot'].mean(), 1))
# kidney['hemo'] = kidney['hemo'].replace('?', round(kidney[kidney['hemo'] != '?']['hemo'].mean(), 1))
# kidney['pcv'] = kidney['pcv'].replace('?', round(kidney[kidney['pcv'] != '?']['pcv'].mean()))
# kidney['wbcc'] = kidney['wbcc'].replace('?', round(kidney[kidney['wbcc'] != '?']['wbcc'].mean()))
# kidney['rbcc'] = kidney['rbcc'].replace('?', round(kidney[kidney['rbcc'] != '?']['rbcc'].mean(), 1))
# kidney['htn'] = kidney['htn'].replace('?', kidney[kidney['htn'] != '?']['htn'].value_counts().idxmax())
# kidney['dm'] = kidney['dm'].replace('?', kidney[kidney['dm'] != '?']['dm'].value_counts().idxmax())
# kidney['cad'] = kidney['cad'].replace('?', kidney[kidney['cad'] != '?']['cad'].value_counts().idxmax())
# kidney['appet'] = kidney['appet'].replace('?', kidney[kidney['appet'] != '?']['appet'].value_counts().idxmax())
# kidney['pe'] = kidney['pe'].replace('?', kidney[kidney['pe'] != '?']['pe'].value_counts().idxmax())
# kidney['ane'] = kidney['ane'].replace('?', kidney[kidney['ane'] != '?']['ane'].value_counts().idxmax())
#
# kidney['rbc'] = pd.factorize(kidney['rbc'])[0]
# kidney['pc'] = pd.factorize(kidney['pc'])[0]
# kidney['pcc'] = pd.factorize(kidney['pcc'])[0]
# kidney['ba'] = pd.factorize(kidney['ba'])[0]
# kidney['htn'] = pd.factorize(kidney['htn'])[0]
# kidney['dm'] = pd.factorize(kidney['dm'])[0]
# kidney['cad'] = pd.factorize(kidney['cad'])[0]
# kidney['appet'] = pd.factorize(kidney['appet'])[0]
# kidney['pe'] = pd.factorize(kidney['pe'])[0]
# kidney['ane'] = pd.factorize(kidney['ane'])[0]
# kidney['class'] = pd.factorize(kidney['class'])[0]
#
# kidney.to_csv('kidney.csv')

# database traffic - needs nothing (maybe little transformation)

traffic = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='TRAFFIC')

traffic.to_csv('traffic.csv')
