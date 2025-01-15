# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 19:42:29 2022

@author: ifons
"""

import numpy as np
import pandas as pd

#Usar reshape para etiquetas
#Supongo arreglo con encabezados
def clasificadorNaiveBayes(datos,etiquetas):
    [_,numY]=etiquetas.shape
    
    [rows,columns]=datos.shape
    
    etiqueta1=etiquetas[0,0]
    
    encabezados=datos[0,:]
    
    [indiceFinVar]=np.where(encabezados==etiqueta1)
    
    prob=np.zeros(numY)
    
    i2=0
    
    for i in etiquetas[0]:
        [_,indice]=np.where(datos==str(i)) #Probar sin str
        
        [datosY]=datos[np.where(datos[:,indice[0]]=='versicolor'),:]
        
        j=0
        
        Px=0.0

        while j < indiceFinVar:
            media=np.mean(datosY[:,j].astype('float64'))
            sigma=np.std(datosY[:,j].astype('float64'))
            
            datosVar=datosY[:,j]
            
           
            print('NuevaVar')
            print(sigma)
            print(media)
            
            for k in datosVar:
                print(k)
                Px=Px+np.log10(1/(sigma*np.sqrt(2*np.pi)))-(1/2)*np.power(((float(k)-media)/sigma),2)
            
            j+=1
            
        [numRowsY,_]=datosY.shape
            
        frecY=numRowsY/(rows-1)
        print('Frec')
        print(rows)
        print(numRowsY)
        print(frecY)
        Px=Px+np.log(frecY)
        
        print('=====')
        print(Px)
        #print(i)        
        #print(prob)
    
        prob[i2]=Px
        
        i2+=1
        
    print('====')
    
    print(prob)
    
    print(np.argmax(prob))
    
print('==========')
#datos=np.array([['G','W','Diabetes','R'],[140,100,'T','F'],[130,50,'T','T'],[90,102,'F','T'],[120,60,'T','T'],[75,90,'F','F'],[110,110,'T','F']])
 
df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

aux=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

datos1 = df.to_numpy()

datos=np.vstack([aux,datos1])

etiquetas=np.array([['species']])
 
clasificadorNaiveBayes(datos,etiquetas.reshape(1,-1))

            
    