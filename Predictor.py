# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 18:21:42 2022

@author: ifons
"""
import numpy as np

#Predictor
#Ejemplo medición de glucosa de 80 personas, me dice quienes pueden tener
def simple_pred(df,theta):
    ftr=df.reshape(1,-1) #No conozco el tamaño del df, lo transformo a una fila
    pred=(ftr>=theta) #Comparo todos los datos contra una variable, genera un vector
    print('feature: ',ftr)
    print('theta: ',theta)
    print('prediction: ',pred)
    
    return pred
    
#
#Llama a predictor, compara las predicciones con el resultado final
#Devuelve un vector con los verdaderos y saca el promedio del renglón
#Verdadero es 1
def simple_acc(ftr,outcome,theta):
    pred=simple_pred(ftr,theta)
    lbl=outcome.reshape(1,-1) #Lo acomodo para que coincida con los datos pred
    cmp=(pred==lbl)
    acc=cmp.mean(axis=1) #Si es fila axis=1
    
    print('outcome: ',lbl)
    print('comparison: ',cmp)
    print('accuracy: ',acc)
    
    return acc

print('==========')
acc=simple_acc(np.array([150., 100., 110., 125.]).reshape(1,-1),
               np.array([True,True,False,False]).reshape(1,-1),
               np.array([120.,130.]).reshape(-1,1))

print('==========')
acc=simple_acc(np.array([150., 100., 110., 125.]).reshape(1,-1),
               np.array([True,False,False,True]).reshape(1,-1),
               np.arange(100,120).reshape(-1,1))




