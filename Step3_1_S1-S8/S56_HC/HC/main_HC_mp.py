# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 20:39:01 2022

@author: chong
"""

import numpy as np
import SWMM_ENV as SWMM_ENV
import datetime
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from swmm_api import read_out_file

env_params={
    'orf':'chaohu_RTC',
    'parm':'./states_yaml/chaohu',
    'GI':False,
    'advance_seconds':30,
    'kf':1,
    'kc':1,
}
env=SWMM_ENV.SWMM_ENV(env_params)

def HC_sample_action(foreaction,observation):
    action=[]
    k=0
    # CC R1
    if observation['CC-storage'] > 1.2:
        a = 1
    elif observation['CC-storage'] < 0.5:
        a = 0
    else:
        if foreaction[k] == 1:
            a = 1
        else:
            a = 0
    action.append(a)
    k+=1
    # CC R2
    if observation['CC-storage'] > 1.4:
        a = 1
    elif observation['CC-storage'] < 0.5:
        a = 0
    else:
        if foreaction[k] == 1:
            a = 1
        else:
            a = 0
    action.append(a)
    k+=1
    # CC S1
    if observation['CC-storage'] > 0.8:
        a = 1
    elif observation['CC-storage'] < 0.5:
        a = 0
    else:
        if foreaction[k] == 1:
            a = 1
        else:
            a = 0
    action.append(a)
    k+=1
    # CC S2
    if observation['CC-storage'] > 1.0:
        a = 1
    elif observation['CC-storage'] < 0.5:
        a = 0
    else:
        if foreaction[k] == 1:
            a = 1
        else:
            a = 0
    action.append(a)
    k+=1
    # JK R1
    if observation['JK-storage'] > 4.2:
        a = 1
    elif observation['JK-storage'] < 1.2:
        a = 0
    else:
        if foreaction[k] == 1:
            a = 1
        else:
            a = 0
    action.append(a)
    k+=1
    # JK R2
    if observation['JK-storage'] > 4.3:
        a = 1
    elif observation['JK-storage'] < 1.2:
        a = 0
    else:
        if foreaction[k] == 1:
            a = 1
        else:
            a = 0
    action.append(a)
    k+=1
    # JK S
    if observation['JK-storage'] > 4.0:
        a = 1
    elif observation['JK-storage'] < 1.2:
        a = 0
    else:
        if foreaction[k] == 1:
            a = 1
        else:
            a = 0
    action.append(a)
    return action

# test 
def test(raine,rainw,i,testid):
    # simulation on given rainfall
    env=SWMM_ENV.SWMM_ENV(env_params)
    test_history = {'time': [], 'state': [], 'action': [], 'reward': [],
                    'F': [], 'C': [], 'RES': [], 'DRES': [], 'mainpip': [], 'pump_flow': [],
                    'capacity':[]}
    storage_states,observation = env.reset(raine,rainw,i,False,testid)
    done, t= False, 0
    test_history['time'].append(t)
    test_history['state'].append(observation)
    foreaction=[0 for _ in range(7)]
    while not done:
        observation = np.array(observation).reshape(1, -1)
        at = HC_sample_action(foreaction,storage_states)
        foreaction = at
        storage_states,observation_new,reward,results,done = env.step(at)
        F,C=results['flooding'],results['CSO']
        Res_tn,DRes_tn=results['Res_tn'],results['DRes_tn']
        pf = results['pump_flow']
        main_flow = results['main_flow']
        capacity = results['capacity']
        t += 1
        
        test_history['time'].append(t)
        test_history['state'].append(observation)
        test_history['action'].append(at)
        test_history['reward'].append(reward)
        test_history['F'].append(F)
        test_history['C'].append(C)
        test_history['RES'].append(Res_tn)
        test_history['DRES'].append(DRes_tn)
        test_history['mainpip'].append(main_flow)
        test_history['pump_flow'].append(pf)
        test_history['capacity'].append(capacity)
        observation = observation_new
        
    return test_history


# RSH test
rainfalle = np.load('./test_rainfall/RSH/east.npy').tolist()
rainfallw = np.load('./test_rainfall/RSH/west.npy').tolist()
for i in range(len(rainfalle)):
    print(i)
    test_his = test(rainfalle[i],rainfallw[i],i,'RSH')
    np.save('./Results/RSH/'+str(i)+'.npy',test_his)


# Real rainfall
rainfalle = np.load('./test_rainfall/RealRain/real.npy').tolist()
rainfallw = np.load('./test_rainfall/RealRain/real.npy').tolist()
for i in range(len(rainfalle)):
    print(i)
    test_his = test(rainfalle[i],rainfallw[i],i,'RR')
    np.save('./Results/RR/'+str(i)+'.npy',test_his)

'''
# RN test
rainfall = np.load('./test_rainfall/2top.npy').tolist()
model.load_model('./model/')
for i in range(len(rainfall)):
    print(i)
    test_his = test(model,rainfall[i],rainfall[i],i)
    np.save('./Results/RN/2top'+str(i)+'.npy',test_his)

rainfall = np.load('./test_rainfall/3top.npy').tolist()
model.load_model('./model/')
for i in range(len(rainfall)):
    print(i)
    test_his = test(model,rainfall[i],rainfall[i],i)
    np.save('./Results/RN/3top'+str(i)+'.npy',test_his)

'''
print('Done')