# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 18:46:29 2022

@author: ifons
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def simple_pred_vec(g, theta):
    
    out=(g>=theta) 
    
    return out

def simple_pred(df, theta):
    
    g=df['Glucose'].values
    pred=simple_pred_vec(g,theta)
    
    return pred

def simple_acc(df, theta):
    
    pred=simple_pred(df,theta)
    lbl=pred.reshape(1,-1) 
    cmp=(pred==lbl)
    acc=cmp.mean(axis=1) 
    
    return acc

def best_theta_loopy(df):
    theta=np.arange(75,200,1)
    
    for i in theta:
        acc=simple_acc(df,i)
    
    best=np.argmax(acc)
    best_theta=theta[best]
    best_accuracy=acc[best]
    
    return np.array([best_theta, best_accuracy])    

df = pd.read_csv('C:\\Users\\ifons\\Documents\\AprendizajeMaquina\\BasicClassification\\diabetes.csv')
df.head()
best_theta_loopy(df)