import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier 
import pickle 

df_train = pd.read_csv('ProcessedTrain_001.csv')
df_test = pd.read_csv('ProcessedTest_001.csv')
df_test=df_test.iloc[:,1:]
df_train=df_train.iloc[:,1:]

x_train=df_train.iloc[:,0:-1]
y_train=df_train.iloc[:,-1]
x_test=df_test.iloc[:,0:-1]
y_test=df_test.iloc[:,-1]

sc= StandardScaler()
sc.fit(x_train)
x_train= sc.transform(x_train)
sc.fit(x_test)
x_test= sc.transform(x_test)

knn= KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train,y_train)

filename = 'knn1.pkl'
pickle.dump(knn, open(filename, 'wb'))
