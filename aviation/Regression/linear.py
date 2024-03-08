import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

from sklearn.linear_model import LinearRegression
import pickle 

def give_Test_engine(Test_no,engine_id):
    "Test_no, is the Sl.no of Test-set"
    "engine_id, is the ID of engine"
    import pandas as pd
    import numpy as np
    df=pd.read_csv("Processed_Test_00{}.csv".format(Test_no))
    test = df[df['ID']==engine_id]
    test = test.drop(columns=['ID'])
    X_test=test.iloc[:,:-1]
    y_test=test.iloc[:,-1]
    return X_test,y_test

def give_Train_engine(Train_no,engine_id):
    "Train_no, is the Sl.no of Train-set"
    "engine_id, is the ID of engine"
    import pandas as pd
    import numpy as np
    df=pd.read_csv("Processed_Train_00{}.csv".format(Train_no))
    train = df[df['ID']==engine_id]
    train = train.drop(columns=['ID'])
    X_train=train.iloc[:,:-1]
    y_train=train.iloc[:,-1]
    return X_train,y_train


df = pd.read_csv('Processed_Train_001.csv')

X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]

# Train Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=False)
lreg = LinearRegression()
lreg.fit(X_train,y_train)

filename = 'lreg1.pkl'
pickle.dump(lreg, open(filename, 'wb'))