from django.contrib import admin
from django.urls import path
from prediction import views

urlpatterns = [
    path('',views.index,name='index'),
    path('index2',views.index2,name='index2'),
    path('RF',views.RF,name='RF'),
    path('CNN',views.CNN,name='CNN'),
    path('DT',views.DT,name='DT'),
    path('KNN',views.KNN,name='KNN'),
    path('rfreg',views.rfreg,name='rfreg'),
    path('NB',views.NB,name='NB'),
    path('poly',views.poly,name='poly'),
    path('SVM',views.SVM,name='SVM'),
    path('reg',views.reg,name='reg'),
    
]
