# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 17:42:35 2022

@author: ifons
"""
import numpy as np
from numpy import pi

"""
a=np.array([1,2,3,4])
a=np.array([(1.5,2,3),(4,5,6)])
a=np.array([[1,2],[3,4]],dtype=complex)
a=np.zeros((3,4))
a=np.one((2,3,4),dtype=np.int16)
a=np.empty((2,3))
a=np.arange(10,30,7)
a=np.linspace(0,2,9)
"""
a=np.linspace(0,2*pi,100)

print(a)
print(a.dtype)

f=np.sin(a)
print(f)