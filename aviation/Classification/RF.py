import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle 

df_train = pd.read_csv('aviation\Classification\ProcessedTrain_001.csv')
df_test = pd.read_csv('aviation\Classification\ProcessedTest_001.csv')
df_test=df_test.iloc[:,1:]
df_train=df_train.iloc[:,1:]

x_train=df_train.iloc[:,0:-1]
y_train=df_train.iloc[:,-1]
x_test=df_test.iloc[:,0:-1]
y_test=df_test.iloc[:,-1]

RF = RandomForestClassifier()
RF.fit(x_train, y_train)

filename = 'aviation\Classification\RF1.pkl'
pickle.dump(RF, open(filename, 'wb'))
