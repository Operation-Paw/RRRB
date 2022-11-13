import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline 
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC




cell_df = pd.read_csv('data.csv')
cell_df.tail()
cell_df.head()
cell_df.shape
cell_df.info


#CLEAN THE DATA
cell_df.count()
cell_df.drop('id',axis=1,inplace=True)
cell_df.drop('Unnamed: 32',axis=1,inplace=True)
# size of the dataframe
len(cell_df)
cell_df.diagnosis.unique()
cell_df['diagnosis'] = cell_df['diagnosis'].map({'M':1,'B':0})
cell_df.head()



#DESCRIBE THE DATA {PLOTTING}
cell_df.describe()
plt.hist(cell_df['diagnosis'])
plt.title('Diagnosis (M=1 , B=0)')
plt.show()



#Observations

#mean values of cell radius, perimeter, area, compactness, concavity and concave points can be used in classification of the cancer. Larger values of these parameters tends to show a correlation with malignant tumors.
#mean values of texture, smoothness, symmetry or fractual dimension does not show a particular preference of one diagnosis over the other. In any of the histograms there are no noticeable large outliers that warrants further cleanup.


features_mean=list(cell_df.columns[1:11])
# split dataframe into two based on diagnosis
dfM=cell_df[cell_df['diagnosis'] ==1]
dfB=cell_df[cell_df['diagnosis'] ==0]






#Stack the data
plt.rcParams.update({'font.size': 12})
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(8,10))
axes = axes.ravel()
for idx,ax in enumerate(axes):
    ax.figure
    binwidth= (max(cell_df[features_mean[idx]]) - min(cell_df[features_mean[idx]]))/50
    ax.hist([dfM[features_mean[idx]],dfB[features_mean[idx]]], bins=np.arange(min(cell_df[features_mean[idx]]), max(cell_df[features_mean[idx]]) + binwidth, binwidth) , alpha=0.5,stacked=True, density = True, label=['M','B'],color=['r','g'])
    ax.legend(loc='upper right')
    ax.set_title(features_mean[idx])
plt.tight_layout()
plt.show()

cell_df.columns



feature_df = cell_df[[ 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']]
x = np.asarray(feature_df)

y = np.asarray(cell_df['diagnosis'])

#TRAINING AND TESTING DATA
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state=4)
x_train.shape
x_test.shape
y_train.shape
y_test.shape
#x[0:5]
#y[0:5]


#modelling the data
classifier = svm.SVC(kernel = 'linear', gamma = 'auto' , C = 2)
classifier.fit(x_train,y_train)
y_predict = classifier.predict(x_test)



#evaluation prediction
print(classification_report(y_test , y_predict))


cell_df.corr().style.background_gradient()

df_new = cell_df.iloc[:,:6]
g = sns.pairplot(df_new,hue='diagnosis')
g.fig.suptitle("diagnosis result", y=1.08)









scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.fit_transform(x_test)
vector = SVC()
vector.fit(X_train_scaled, y_train)
print(f"Support vector machine training set accuracy: {format(vector.score(X_train_scaled, y_train), '.4f')} ")
print(f"Support vector machine testing set accuracy: {format(vector.score(X_test_scaled, y_test), '.4f')} ")


