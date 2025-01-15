# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 18:09:08 2022

@author: ifons
"""

import numpy as np

a=np.array([1,2,3,4]).reshape(1,-1)
b=np.array([1,3,6]).reshape(-1,1)

c=(a==b)

print(c)

d=a*b
print(d)

e=a+b
print(e)

f=a>b
print(f)

"""
a1=np.array([(1,3,5),(4,7,10)]) #No es válido
b1=np.array([3,0.5,2]).reshape(-1,1)

c1=a1*b1

print(c1)
"""-