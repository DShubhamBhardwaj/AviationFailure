import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
import pickle 

df_train = pd.read_csv('ProcessedTrain_001.csv')
df_test = pd.read_csv('ProcessedTest_001.csv')
df_test=df_test.iloc[:,1:]
df_train=df_train.iloc[:,1:]

x_train=df_train.iloc[:,0:-1]
y_train=df_train.iloc[:,-1]
x_test=df_test.iloc[:,0:-1]
y_test=df_test.iloc[:,-1]


sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

classifierNB = GaussianNB()
classifierNB.fit(X_train, y_train)

filename = 'nb1.pkl'
pickle.dump(classifierNB, open(filename, 'wb'))
