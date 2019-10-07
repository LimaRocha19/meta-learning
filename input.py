import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas import ExcelWriter
from pandas import ExcelFile

# database abalone - needs to transform values

# abalone = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='ABALONE')
#
# abalone['Sex'] = abalone['Sex'].replace('I', 0)
# abalone['Sex'] = abalone['Sex'].replace('M', 1)
# abalone['Sex'] = abalone['Sex'].replace('F', 2)
#
# abalone.to_csv('abalone.csv')

# database adult - needs to replace missing value and transform values

# adult = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='ADULT')
#
# adult['workclass'] = adult['workclass'].replace('?', adult['workclass'].value_counts().idxmax())
# adult['occupation'] = adult['occupation'].replace('?', adult['occupation'].value_counts().idxmax())
# adult['native-country'] = adult['native-country'].replace('?', adult['native-country'].value_counts().idxmax())
#
# adult.to_csv('adult.csv')

# database australian - needs to transform values

# database drugs - needs to transform values

# database fertility - needs to transform values

# database german - needs nothing (maybe little transformation)

# database glass - needs nothing (maybe little transformation)

# database heart - needs nothing (maybe little transformation)

# database ionosphere - needs to transform values

# database pendigits - needs nothing (maybe little transformation)

# database phishing - needs nothing (maybe little transformation)

# database failures - needs nothing (maybe little transformation)

# database shuttle - needs nothing (maybe little transformation)

# database spam - needs nothing (maybe little transformation)

# database wdbc - needs to transform values

# database wifi - needs nothing (maybe little transformation)

# database wine - needs nothing (maybe little transformation)

# database zoo - needs to transform values

# database breast - needs to transform values

# database stability - needs to transform values

# database trip - needs nothing (maybe little transformation)

# database student - needs to transform values

# database leaf - needs nothing (maybe little transformation)

# database kidney - needs to replace missing value and transform values

# kidney = pd.read_excel('Bases_Consolidadas.xlsx', sheet_name='KIDNEY')
#
# kidney['age'] = kidney['age'].replace('?', round(kidney[kidney['age'] != '?']['age'].mean()))
# kidney['bp'] = kidney['bp'].replace('?', round(kidney[kidney['bp'] != '?']['bp'].mean()))
# kidney['sg'] = kidney['sg'].replace('?', kidney['sg'].value_counts().idxmax())
# kidney['al'] = kidney['al'].replace('?', kidney['al'].value_counts().idxmax())
# kidney['su'] = kidney['su'].replace('?', kidney['su'].value_counts().idxmax())
# kidney['rbc'] = kidney['rbc'].replace('?', kidney['rbc'].value_counts().idxmax())
# kidney['pc'] = kidney['pc'].replace('?', kidney['pc'].value_counts().idxmax())
# kidney['pcc'] = kidney['pcc'].replace('?', kidney['pcc'].value_counts().idxmax())
# kidney['ba'] = kidney['ba'].replace('?', kidney['ba'].value_counts().idxmax())
# kidney['bgr'] = kidney['bgr'].replace('?', round(kidney[kidney['bgr'] != '?']['bgr'].mean()))
# kidney['bu'] = kidney['bu'].replace('?', round(kidney[kidney['bu'] != '?']['bu'].mean()))
# kidney['sc'] = kidney['sc'].replace('?', round(kidney[kidney['sc'] != '?']['sc'].mean(), 1))
# kidney['sod'] = kidney['sod'].replace('?', round(kidney[kidney['sod'] != '?']['sod'].mean()))
# kidney['pot'] = kidney['pot'].replace('?', round(kidney[kidney['pot'] != '?']['pot'].mean(), 1))
# kidney['hemo'] = kidney['hemo'].replace('?', round(kidney[kidney['hemo'] != '?']['hemo'].mean(), 1))
# kidney['pcv'] = kidney['pcv'].replace('?', round(kidney[kidney['pcv'] != '?']['pcv'].mean()))
# kidney['wbcc'] = kidney['wbcc'].replace('?', round(kidney[kidney['wbcc'] != '?']['wbcc'].mean()))
# kidney['rbcc'] = kidney['rbcc'].replace('?', round(kidney[kidney['rbcc'] != '?']['rbcc'].mean(), 1))
# kidney['htn'] = kidney['htn'].replace('?', kidney['htn'].value_counts().idxmax())
# kidney['dm'] = kidney['dm'].replace('?', kidney['dm'].value_counts().idxmax())
# kidney['cad'] = kidney['cad'].replace('?', kidney['cad'].value_counts().idxmax())
# kidney['appet'] = kidney['appet'].replace('?', kidney['appet'].value_counts().idxmax())
# kidney['pe'] = kidney['pe'].replace('?', kidney['pe'].value_counts().idxmax())
# kidney['ane'] = kidney['ane'].replace('?', kidney['ane'].value_counts().idxmax())
#
# kidney.to_csv('kidney.csv')

# database traffic - needs nothing (maybe little transformation)
