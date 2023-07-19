# -*- coding: utf-8 -*-

"""

Created on Thu Sep 22 16:06:17 2022



@author: chongloc

"""

import numpy as np

import Rainfall_data as RD





'''

parameters range:

    A:21-35

    C:0.939-1.2

    P:1-5

    n:0.86-0.96

    b:16-22

    r:0.3-0.8

'''



raindata=[]

delta,dura = 1,120





for _ in range(1000):

    A = np.random.randint(21,35)

    C = np.random.randint(93,120)/100

    P = np.random.randint(1,5)

    b = np.random.randint(16,22)

    n = np.random.randint(86,96)/100

    r = np.random.randint(3,8)/10 

    

    para_tuple = (A,C,n,b,r,P,delta,dura)

    tem = RD.Chicago_icm(para_tuple)

    raindata.append(tem)

    

np.save('training_raindata.npy',np.array(raindata))





'''

Test rainfall parameters:

    A,C,P,n,b,r

    (21,0.939,5,0.86,16,0.3)

    (28,0.95,,0.90,19,0.4)

    (35,1.1,,0.96,22,0.5)

    (21,0.939,,0.86,16,0.6)

    (28,0.95,,0.90,19,0.7)

    (35,1.1,,0.96,22,0.8)

'''



raindata=[]

parameters=[(21,0.939,0.86,16,0.3,1),

            (28,0.95,0.90,19,0.4,3),

            (35,1.1,0.96,22,0.5,5),

            (21,0.939,0.86,16,0.6,1),

            (28,0.95,0.90,19,0.7,3),

            (35,1.1,0.96,22,0.8,5)]



for p in parameters:

    A = np.random.randint(21,35)

    C = np.random.randint(93,120)/100

    P = np.random.randint(1,5)

    b = np.random.randint(16,22)

    n = np.random.randint(86,96)/100

    r = np.random.randint(3,8)/10 

    

    para_tuple = (A,C,n,b,r,P,delta,dura)

    tem = RD.Chicago_icm(para_tuple)

    raindata.append(tem)

    

np.save('test_raindata.npy',np.array(raindata))

