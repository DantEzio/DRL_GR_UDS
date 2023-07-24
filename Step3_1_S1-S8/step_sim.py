import pandas as pd
from swmm_api.input_file import read_inp_file
from swmm_api.input_file.section_labels import *
from swmm_api.input_file.sections import *
from swmm_api.input_file.helpers import *
from swmm_api.input_file.sections.others import TimeseriesData
from swmm_api.input_file.section_labels import TIMESERIES
import os
os.environ['CONDA_DLL_SEARCH_MODIFICATION_ENABLE']="1"
import numpy as np
import shutil
import yaml
from pyswmm import Simulation,Links,Nodes,RainGages,SystemStats
import matplotlib.pyplot as plt


Interval_t=300

def get_step(sim,config,params,tn,Qn,sflooding,sCSO):
    #获取模拟结果
    nodes = Nodes(sim)
    links = Links(sim)
    rgs = RainGages(sim)
    sys = SystemStats(sim)
    #obtain states and reward term by yaml (config)
    states = []
    for _temp in config["states"]:
        if _temp[1] == 'depthN':
            states.append(nodes[_temp[0]].depth)
        elif _temp[1] == 'flow':
            states.append(links[_temp[0]].flow)
        elif _temp[1] == 'inflow':
            states.append(nodes[_temp[0]].total_inflow)
        else:
            states.append(rgs[_temp[0]].rainfall)
    
    #获取reward
    #reward分为2个部分，Res和DRes，其中Res通过计算SevC和SevF获取，DRes通过计算干管节点流量获取
    #均需要通过流量乘时间间隔计算
    flooding,CSO,CSOtem=0,0,0
    Qtw=0
    tem_sevC,tem_sevF=0,0
    tem_dres=0
    
    
    for _temp in config['reward_targets']:
        if _temp[0] == 'sevF':
            for n in nodes:
                deltt=(n.statistics['flooding_duration']-tn[n.nodeid])*60 #minutes
                tem_sevF+=n.flooding*deltt/Interval_t
                tn[n.nodeid]=n.statistics['flooding_duration']
            
        elif _temp[0] == 'sevC':
            tem_sevC += nodes[_temp[1]].total_inflow

        elif _temp[0] == 'DRes': 
            tem_dres += np.square(nodes[_temp[1]].total_inflow/5-Qn[_temp[1]]/5) #minutes
            Qn[_temp[1]] = nodes[_temp[1]].total_inflow
            
        elif _temp[0] == 'Flooding':
            #累积的flooding
            flooding += sys.routing_stats[_temp[1]] - sflooding
            sflooding = sys.routing_stats[_temp[1]]
        
        elif  _temp[0] == 'CSO':
            #累积的CSO
            CSOtem += nodes[_temp[1]].cumulative_inflow
            
    Qtw = (sys.routing_stats['dry_weather_inflow']
            +sys.routing_stats['wet_weather_inflow']
            +sys.routing_stats['groundwater_inflow']
            +sys.routing_stats['II_inflow'])
    
    Res_tn = 1/(1+params['kc']*((tem_sevC)/(Qtw))+params['kf']*((tem_sevF)/(Qtw)))
    DRes_tn = 1/(1+(tem_dres)/(Qtw))
    CSO = CSOtem - sCSO
    sCSO = CSOtem
    rewards = Res_tn+DRes_tn
    
    #获取干管流量结果
    mainpip=['WS02006251WS02006249','WS02006235WS02006234','WS02006229WS02006228']
    main_flow=[]
    for pip in mainpip:
        main_flow.append(links[pip].flow)
    return Qn,tn,states,rewards,sflooding,sCSO,Res_tn,DRes_tn,main_flow

def simfile(Scenario,file,rainid,config,params):
    sim=Simulation(file)
    #sim.execute()
    sim.start()
    sim._model.swmm_stride(300)
    test_history = {'time':[] ,'state': [], 'action': [], 'reward': [], 
                    'F':[], 'C':[], 'RES':[], 'DRES':[], 'mainpip':[]}
    cur_flooding,cur_CSO=0,0
    nodes = Nodes(sim)
    links = Links(sim)
    rgs = RainGages(sim)
    sys = SystemStats(sim)    
    tn={}
    for n in nodes:
        tn[n.nodeid]=n.statistics['flooding_duration']
    Qn={}
    for _temp in config['reward_targets']:
        if _temp[0] == 'DRes': 
            Qn[_temp[1]] = nodes[_temp[1]].total_inflow
    for t in range(95*5):
        sim._model.swmm_stride(60)
        Qn,tn,observation,reward,cur_flooding,cur_CSO,Res_tn,DRes_tn,main_flow=get_step(sim,config,params,tn,Qn,cur_flooding,cur_CSO)
        test_history['time'].append(t)
        test_history['state'].append(observation)
        test_history['reward'].append(reward)
        test_history['F'].append(cur_flooding)
        test_history['C'].append(cur_CSO)
        test_history['RES'].append(Res_tn)
        test_history['DRES'].append(DRes_tn)
        test_history['mainpip'].append(main_flow)
    sim._model.swmm_end()
    sim._model.swmm_close()
    np.save('./'+Scenario+'/Results/'+str(rainid)+'.npy',test_history)