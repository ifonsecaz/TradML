# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 17:57:08 2022

@author: ifons
"""
import numpy as np

a= np.arange(6)
print("a \n",a)

b=np.arange(12).reshape(4,3)
print("b \n",b)

c=np.arange(24).reshape(2,3,4)
print("c \n",c)

c=c.reshape(3,2,4)
print("c \n",c)

c=c.reshape(3,-1,4)
print("c \n",c)

c=c.reshape(3,-1)
print("c \n",c)

c=c.reshape(-1,3)
print("c \n",c)

c=c.reshape(-1)
print("c \n",c)

c=c.reshape(1,-1)
print("c \n",c)

c=c.reshape(-1,1)
print("c \n",c)

####

d= np.zeros((2,2))

e=d
d[0,0]=2
print(d)
print(e)

###Lista python
f=[0,0]
g=f
f[1]=2
print(f)
print(g)








