"# RRRB" 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
cell_df = pd.read_csv('Outbreak_240817.csv')
#cell_df.tail()
#cell_df.head()
#cell_df.shape
#cell_df.count()
cell_df['status'].value_counts()
confirmed_df = cell_df[cell_df["status"]==17002][0:1000]
denied_df = cell_df[cell_df["status"]==6][0:3]
#help(confirmed_df.plot)
#axes = confirmed_df.plot(kind ="scatter", x ='Confirmed', y ='Denied', color ='blue', lable ='bluetounge')
#denied_df.plot(kind = 'scatter', x = 'Confirmed', y = 'Denied', color = 'red', lable = 'influenza', ax = axes)

cell_df = cell_df[pd.to_numeric(cell_df['status'], errors='coerce').notnull()]
cell_df['status'] = cell_df['status'].astype('int')
cell_df['source'] = cell_df['source'].astype('int')
cell_df['region'] = cell_df['region'].astype('int')
cell_df['country'] = cell_df['country'].astype('int')
cell_df['admin1'] = cell_df['admin1'].astype('int')
cell_df['localityName'] = cell_df['localityName'].astype('int')
cell_df['localityQuality'] = cell_df['localityQuality'].astype('int')
cell_df['observationDate'] = cell_df['observationDate'].astype('int')
cell_df['reportingDate'] = cell_df['reportingDate'].astype('int')
cell_df['disease'] = cell_df['disease'].astype('int')
cell_df['serotypes'] = cell_df['serotypes'].astype('int')
cell_df['speciesDescription'] = cell_df['speciesDescription'].astype('int')
cell_df['humansGenderDesc'] = cell_df['humansGenderDesc'].astype('int')
feature_df = cell_df[[ 'source', 'latitude', 'longitude', 'region', 'country', 'admin1',
       'localityName', 'localityQuality', 'observationDate', 'reportingDate',
        'disease','serotypes', 'speciesDescription', 'sumAtRisk',
       'sumCases', 'sumDeaths', 'sumDestroyed', 'sumSlaughtered',
       'humansGenderDesc', 'humansAge', 'humansAffected', 'humansDeaths']]
x = np.asarray(feature_df)
y = np.asarray(cell_df['status'])
#x[0:5]
#y[0:5]
#TRAINING AND TESTING DATA
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state=4)
x_train.shape

