from django.shortcuts import render
import pickle
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn import metrics
from keras.models import load_model
from plotly.graph_objs import Bar, Scatter
import json,plotly
from .models import rul
# Create your views here.
def index(request):
    return render(request,'index.html')
def index2(request):
    return render(request,'index2.html')
def RF(request):
    if request.POST:
        example = [8,642.54,1580.89,1400.89,553.59,2388.05,47.21,522.09,2388.06,8.4213,393,39.05,23.3224]
        li_para = ['cycle','s2', 's3','s4','s7','s8','s11','s12','s13','s15','s17','s20','s21']
        for ind, i in enumerate(li_para):
            value = request.POST.get(i)
            example[ind] = value
        example = pd.DataFrame(example)
        example = example.transpose()
        filename = 'Classification/RF1.pkl'
        loaded_model = pickle.load(open(filename, 'rb'))
        result = loaded_model.predict(example)
        dict = {'var': result}
    else:
        dict = {'var': None}
    return render(request,'RF.html',dict)

    
def DT(request):
    if request.POST:
        example = [8,642.54,1580.89,1400.89,553.59,2388.05,47.21,522.09,2388.06,8.4213,393,39.05,23.3224]
        li_para = ['cycle','s2', 's3','s4','s7','s8','s11','s12','s13','s15','s17','s20','s21']
        for ind, i in enumerate(li_para):
            value = request.POST.get(i)
            example[ind] = value
        example = pd.DataFrame(example)
        example = example.transpose()
        filename = 'Classification/DT1.pkl'
        loaded_model = pickle.load(open(filename, 'rb'))
        result = loaded_model.predict(example)
        dict = {'var': result}
    else:
        dict = {'var': None}
    return render(request,'DT.html',dict)
def KNN(request):
    if request.POST:
        example = [8,642.54,1580.89,1400.89,553.59,2388.05,47.21,522.09,2388.06,8.4213,393,39.05,23.3224]
        li_para = ['cycle','s2', 's3','s4','s7','s8','s11','s12','s13','s15','s17','s20','s21']
        for ind, i in enumerate(li_para):
            value = request.POST.get(i)
            example[ind] = value
        example = pd.DataFrame(example)
        example = example.transpose()
        filename = 'Classification/knn1.pkl'
        loaded_model = pickle.load(open(filename, 'rb'))
        result = loaded_model.predict(example)
        dict = {'var': result}
    else:
        dict = {'var': None}
    return render(request,'KNN.html',dict)
def NB(request):
    if request.POST:
        example = [8,642.54,1580.89,1400.89,553.59,2388.05,47.21,522.09,2388.06,8.4213,393,39.05,23.3224]
        li_para = ['cycle','s2', 's3','s4','s7','s8','s11','s12','s13','s15','s17','s20','s21']
        for ind, i in enumerate(li_para):
            value = request.POST.get(i)
            example[ind] = value
        example = pd.DataFrame(example)
        example = example.transpose()
        filename = 'Classification/nb1.pkl'
        loaded_model = pickle.load(open(filename, 'rb'))
        result = loaded_model.predict(example)
        dict = {'var': result}
    else:
        dict = {'var': None}
    return render(request,'NB.html',dict)
def SVM(request):
    if request.POST:
        example = [8,642.54,1580.89,1400.89,553.59,2388.05,47.21,522.09,2388.06,8.4213,393,39.05,23.3224]
        li_para = ['cycle','s2', 's3','s4','s7','s8','s11','s12','s13','s15','s17','s20','s21']
        for ind, i in enumerate(li_para):
            value = request.POST.get(i)
            example[ind] = value
        example = pd.DataFrame(example)
        example = example.transpose()
        filename = 'Classification/svm1.pkl'
        loaded_model = pickle.load(open(filename, 'rb'))
        result = loaded_model.predict(example)
        dict = {'var': result}
    else:
        dict = {'var': None}
    return render(request,'SVM.html',dict)

def reg(request):
    if request.POST:
        example = [1,8,642.54,1580.89,1400.89,553.59,2388.05,47.21,522.09,2388.06,8.4213,393,39.05,23.3224]
        li_para = ['id','cycle','s2', 's3','s4','s7','s8','s11','s12','s13','s15','s17','s20','s21']
        for ind, i in enumerate(li_para):
            value = request.POST.get(i)
            example[ind] = value
        print(example)
        example = pd.DataFrame(example)
        example = example.transpose()
        filename = 'Regression/lreg1.pkl'
        loaded_model = pickle.load(open(filename, 'rb'))
        result = loaded_model.predict(example)
        dict = {'var': result}
    else:
        dict = {'var': None}
    return render(request,'reg.html',dict)
    

def poly(request):
    from sklearn.preprocessing import PolynomialFeatures
    if request.POST:
        example = [1,8,642.54,1580.89,1400.89,553.59,2388.05,47.21,522.09,2388.06,8.4213,393,39.05,23.3224]
        li_para = ['id','cycle','s2', 's3','s4','s7','s8','s11','s12','s13','s15','s17','s20','s21']
        for ind, i in enumerate(li_para):
            value = request.POST.get(i)
            example[ind] = value
        print(example)
        example = pd.DataFrame(example)
        example = example.transpose()
        polynomial_features= PolynomialFeatures(degree=4)
        x_poly = polynomial_features.fit_transform(example) 
        filename = 'Regression/poly1.pkl'
        loaded_model = pickle.load(open(filename, 'rb'))
        result = loaded_model.predict(x_poly)
        dict = {'var': result}
    else:
        dict = {'var': None}
    return render(request,'poly.html',dict)

def rfreg(request):
    if request.POST:
        example = [1,8,642.54,1580.89,1400.89,553.59,2388.05,47.21,522.09,2388.06,8.4213,393,39.05,23.3224]
        li_para = ['id','cycle','s2', 's3','s4','s7','s8','s11','s12','s13','s15','s17','s20','s21']
        for ind, i in enumerate(li_para):
            value = request.POST.get(i)
            example[ind] = value
        print(example)
        example = pd.DataFrame(example)
        example = example.transpose()
        filename = 'Regression/rfreg1.pkl'
        loaded_model = pickle.load(open(filename, 'rb'))
        result = loaded_model.predict(example)
        dict = {'var': result}
    else:
        dict = {'var': None}
    return render(request,'rfreg.html',dict)

##CNN Starts HERE 
win_length = 25   ######### Sliding Window Length
feature_num = 13  ######### Total number of features

def Data_format_conversion(Train_no,engine_id):    

    import pandas as pd
    import numpy as np
    df=pd.read_csv("C:/Users/D R Bhardwaj/Desktop/project/aviation/Regression/Processed_Train_00{}.csv".format(Train_no))
    df = df[df['ID']==engine_id]
    df = df.drop(columns=['ID'])

    ################################## Scalling the DATA
    scaler=MinMaxScaler()
    df = scaler.fit_transform(df)

    ################################    Getting into training shape with slidingwindow
    features = df[:,0:-1]
    target = df[:,-1]



    ts_generator = TimeseriesGenerator(features,target,length=win_length,sampling_rate=1,batch_size=1)

    ################################ Changing the shape of input to (no of smaples,window_length,features)
    X=[]
    y=[]
    for i in range(len(ts_generator)):
        x_temp, y_temp = ts_generator[i]
        X.append(x_temp.reshape(x_temp.shape[1],x_temp.shape[2],1))
        y.append(y_temp)

    X=np.array(X)  
    y=np.array(y)  
    
    return(X,y,scaler,features)


def CNN(request):
    if True:
        Train_no=1
        if request.POST:
            engine_id= int(request.POST.get('id'))
            print(engine_id)
        else:
            engine_id=1
        X,y,scaler,features=Data_format_conversion(Train_no,engine_id) 
        model = load_model('Regression/cnn1.h5')
        prediction=model.predict(X) ######### prediction on trained data
        rev_trans =pd.concat([pd.DataFrame(features[win_length:]),pd.DataFrame(prediction)],axis=1)
        rev_trans = scaler.inverse_transform(rev_trans)######## Transforming back to original scale
        rev_trans =pd.DataFrame(rev_trans)
        df=pd.read_csv("C:/Users/D R Bhardwaj/Desktop/project/aviation/Regression/Processed_Train_00{}.csv".format(Train_no))
        df = df[df['ID']==engine_id]
        df_actual = df.drop(columns=['ID'])
        # plt.plot(df_actual['Cycle'][win_length:],df_actual['RUL'][win_length:])
        # plt.plot(rev_trans[0],rev_trans[13])
        # plt.ylabel('RUL')
        # plt.xlabel('CYCLE')
        # plt.title('Engine No is: {}'.format(engine_id))
        # plt.legend([ 'Actual','Prediction'], loc='upper right')
        # plt.show()
        graphs = [
        {
            'data': [
                Scatter(
                    x=df_actual['Cycle'][win_length:],
                    y=df_actual['RUL'][win_length:]
                )
            ],
            'data2': [
            Scatter(
                x = rev_trans[0],
                y = rev_trans[13]
            )
        ], 
            'layout': {
                
                'yaxis': {
                    'title': "RUL"
                },
                'xaxis': {
                    'title': "Cycle"
                }
            }
            }
        ] 
        ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]  
        graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    # else:
    dict = {'graphJSON': graphJSON,
            'ids' : ids,
            'engine': engine_id}
    return render(request,'cnn.html',dict)


def ANN(request):
    return render(request,'ann.html')    